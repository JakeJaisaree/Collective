import os, json, base64, tempfile, uuid, time, logging
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, auth, firestore

import stripe
from openai import OpenAI

# -----------------------------
# Environment / Config
# -----------------------------
load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://aquarian-8c213.web.app")
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    FRONTEND_URL,
] + [o.strip() for o in os.getenv("CORS_EXTRA", "").split(",") if o.strip()]

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET:
    raise RuntimeError("Stripe keys required.")

PRICE_CENTS = int(os.getenv("PRICE_CENTS", "500"))
CURRENCY = os.getenv("CURRENCY", "usd")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # cheap/fast default

FIREBASE_KEY_B64 = os.getenv("FIREBASE_KEY_B64")
FIREBASE_KEY_PATH = None
if FIREBASE_KEY_B64:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(base64.b64decode(FIREBASE_KEY_B64))
        FIREBASE_KEY_PATH = f.name
else:
    FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "config/aquarian_key.json")

# -----------------------------
# Init
# -----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aquarian")

cred = credentials.Certificate(FIREBASE_KEY_PATH)
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

stripe.api_key = STRIPE_SECRET_KEY
oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Aquarian")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer()

# -----------------------------
# Rate limit (simple, per-UID)
# -----------------------------
# Token bucket: 5 burst, 1 token/sec
rate_state: Dict[str, Dict[str, float]] = {}
def ratelimit(uid: str, bucket="chat", capacity=5, refill_rate=1.0):
    now = time.time()
    st = rate_state.setdefault(uid, {}).setdefault(bucket, {"tokens": capacity, "ts": now})
    # refill
    delta = now - st["ts"]
    st["tokens"] = min(capacity, st["tokens"] + delta * refill_rate)
    st["ts"] = now
    if st["tokens"] < 1:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")
    st["tokens"] -= 1

# -----------------------------
# Basis enums & models
# -----------------------------
class NodeType(str, Enum):
    Object = "Object"
    Configuration = "Configuration"

class Quality(str, Enum):
    Content = "U"
    Stress = "▲"

class Action(str, Enum):
    Connect = "+"
    Disconnect = "-"
    Naught = "..."

class User(BaseModel):
    uid: str
    email: str = ""
    pro: bool = False

class Node(BaseModel):
    node_id: str
    node_type: NodeType
    quality: Quality
    energy: int = Field(ge=1, le=5)
    last_updated: str

class Connection(BaseModel):
    source_node_id: str
    target_node_id: str
    action: Action
    weight: float
    effect: Optional[int] = None  # Φ
    context: Optional[str] = ""
    created: Optional[str] = None
    updated: Optional[str] = None

class UpsertNodeRequest(BaseModel):
    node_type: NodeType
    quality: Quality
    energy: int = Field(ge=1, le=5)

class UpsertConnRequest(BaseModel):
    source_node_id: str
    target_node_id: str
    action: Action
    weight: float
    context: Optional[str] = ""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    actions: Optional[dict] = None

# -----------------------------
# Auth helper
# -----------------------------
async def get_current_user(token: HTTPAuthorizationCredentials = Depends(bearer)) -> User:
    if not token or not token.credentials:
        raise HTTPException(status_code=401, detail="Missing auth token.")
    try:
        decoded = auth.verify_id_token(token.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user_ref = db.collection("users").document(decoded["uid"])
    snap = user_ref.get()
    data = snap.to_dict() if snap.exists else {}
    return User(uid=decoded["uid"], email=decoded.get("email", ""), pro=data.get("pro", False))

# -----------------------------
# Helpers: Basis logic
# -----------------------------
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def compute_phi(user_uid: str, node_ids: List[str]) -> int:
    total = 0
    nodes_coll = db.collection("users").document(user_uid).collection("nodes")
    for nid in node_ids:
        snap = nodes_coll.document(nid).get()
        if not snap.exists:
            continue
        n = snap.to_dict() or {}
        energy = int(n.get("energy", 0))
        q = n.get("quality")
        if q == Quality.Content.value:
            total += energy
        elif q == Quality.Stress.value:
            total -= energy
    return total

def validate_transition(prev: Optional[Action], new: Action):
    if prev is None or prev == new or new == Action.Naught:
        return
    if (prev == Action.Connect and new == Action.Disconnect) or (prev == Action.Disconnect and new == Action.Connect):
        raise HTTPException(status_code=400, detail="Invalid transition: must pass through '...' before switching between '+' and '-'.")

def get_or_create_customer(uid: str, email: str = "") -> str:
    user_ref = db.collection("users").document(uid)
    snap = user_ref.get()
    data = snap.to_dict() if snap.exists else {}
    if data and data.get("stripeCustomerId"):
        return data["stripeCustomerId"]
    cust = stripe.Customer.create(metadata={"uid": uid}, email=email or None)
    user_ref.set({"stripeCustomerId": cust.id}, merge=True)
    return cust.id

def persist_message(uid: str, role: str, content: str):
    db.collection("users").document(uid).collection("messages").add({
        "role": role, "content": content, "ts": datetime.utcnow().isoformat()
    })

def fetch_recent_messages(uid: str, limit: int = 15) -> List[Dict]:
    msgs = db.collection("users").document(uid).collection("messages") \
        .order_by("ts", direction=firestore.Query.DESCENDING).limit(limit).stream()
    arr = list(msgs)
    arr = [m.to_dict() for m in reversed(arr)]
    return arr

# -----------------------------
# Routes
# -----------------------------
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}

@app.post("/user/me", tags=["User"])
async def create_or_get_user(user=Depends(get_current_user)):
    user_ref = db.collection("users").document(user.uid)
    snap = user_ref.get()
    if not snap.exists:
        user_ref.set({"email": user.email, "pro": False})
        return {"uid": user.uid, "email": user.email, "pro": False}
    d = snap.to_dict() or {}
    return {"uid": user.uid, "email": d.get("email", user.email), "pro": d.get("pro", False)}

@app.post("/network/node", tags=["Network"])
async def upsert_node(req: UpsertNodeRequest, user=Depends(get_current_user)):
    nodes = db.collection("users").document(user.uid).collection("nodes")
    query = nodes.where("node_type", "==", req.node_type.value).where("quality", "==", req.quality.value).limit(1)
    hits = list(query.stream())
    now = datetime.utcnow().isoformat()
    if hits:
        doc = hits[0]
        doc.reference.update({"energy": clamp(req.energy, 1, 5), "last_updated": now})
        return doc.reference.get().to_dict()
    node_id = f"N{uuid.uuid4().hex[:8]}"
    node = {
        "node_id": node_id,
        "node_type": req.node_type.value,
        "quality": req.quality.value,
        "energy": clamp(req.energy, 1, 5),
        "last_updated": now,
    }
    nodes.document(node_id).set(node)
    return node

@app.get("/network", tags=["Network"])
async def fetch_network(user=Depends(get_current_user)):
    base = db.collection("users").document(user.uid)
    nodes = [n.to_dict() for n in base.collection("nodes").stream()]
    conns = [c.to_dict() for c in base.collection("connections").stream()]
    return {"nodes": nodes, "connections": conns}

@app.post("/network/connection", tags=["Network"])
async def upsert_connection(req: UpsertConnRequest, user=Depends(get_current_user)):
    conns = db.collection("users").document(user.uid).collection("connections")
    query = conns.where("source_node_id", "==", req.source_node_id) \
                 .where("target_node_id", "==", req.target_node_id).limit(1)
    hits = list(query.stream())
    now = datetime.utcnow().isoformat()
    phi = compute_phi(user.uid, [req.source_node_id, req.target_node_id])

    if hits:
        doc = hits[0]
        cur = doc.to_dict() or {}
        prev = None
        try:
            prev = Action(cur.get("action"))
        except Exception:
            prev = None
        validate_transition(prev, req.action)
        doc.reference.update({
            "action": req.action.value,
            "weight": float(req.weight),
            "context": req.context or "",
            "effect": int(phi),
            "updated": now
        })
        return doc.reference.get().to_dict()

    conn_id = f"C{uuid.uuid4().hex[:8]}"
    conn = {
        "source_node_id": req.source_node_id,
        "target_node_id": req.target_node_id,
        "action": req.action.value,
        "weight": float(req.weight),
        "effect": int(phi),
        "context": req.context or "",
        "created": now,
        "updated": now,
    }
    conns.document(conn_id).set(conn)
    return conn

@app.get("/pro/status", tags=["Pro"])
async def check_pro(user=Depends(get_current_user)):
    if not user.pro:
        raise HTTPException(status_code=402, detail="Pro required")
    return {"pro": True}

@app.post("/create-payment-intent", tags=["Stripe"])
async def create_payment_intent(user=Depends(get_current_user)):
    try:
        cust = get_or_create_customer(user.uid, user.email)
        intent = stripe.PaymentIntent.create(
            amount=PRICE_CENTS,
            currency=CURRENCY,
            payment_method_types=["card"],
            customer=cust,
            metadata={"uid": user.uid},
            description="Aquarian Pro",
        )
        return {"clientSecret": intent.client_secret}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stripe/webhook", tags=["Stripe"])
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature") or request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Webhook verification failed")

    if event.get("type") == "payment_intent.succeeded":
        intent = event["data"]["object"]
        uid = (intent.get("metadata") or {}).get("uid")
        if uid:
            ref = db.collection("users").document(uid)
            snap = ref.get()
            if snap.exists:
                data = snap.to_dict() or {}
                if not data.get("pro"):
                    ref.update({"pro": True})
    return {"status": "ok"}

# -----------------------------
# Chat with action extraction
# -----------------------------
SYSTEM_MSG = """You are Aquarian, a helpful assistant. 
In addition to your natural-language reply, emit a STRICT JSON block under the key "actions" describing graph updates.

Schema:
{
  "nodes": [
    {"node_type":"Object|Configuration", "quality":"U|▲", "energy":1-5}
  ],
  "connections": [
    {"source_node_id":"<id>", "target_node_id":"<id>", "action":"+|-|...", "weight":0.0-5.0, "context":"<optional>"}
  ]
}

Output format (IMPORTANT):
Return a single JSON object: {"reply":"<string>","actions":{"nodes":[...],"connections":[...]}} 
Do not add explanations outside JSON.
If no updates, set empty arrays.
"""

def parse_actions(blob: str) -> ChatResponse:
    # Expect strict JSON. Fall back to heuristic extraction if necessary.
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # Try to extract the last {...} block
        start = blob.find("{")
        end = blob.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ChatResponse(reply=blob, actions={"nodes": [], "connections": []})
        try:
            data = json.loads(blob[start:end+1])
        except Exception:
            return ChatResponse(reply=blob, actions={"nodes": [], "connections": []})

    reply = data.get("reply", "")
    actions = data.get("actions") or {"nodes": [], "connections": []}
    # Normalize arrays
    actions.setdefault("nodes", [])
    actions.setdefault("connections", [])
    return ChatResponse(reply=reply, actions=actions)

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    if not oai:
        raise HTTPException(status_code=503, detail="OpenAI not configured.")

    ratelimit(user.uid, bucket="chat")

    # persist user msg
    persist_message(user.uid, "user", req.message)

    # prepare context
    history = fetch_recent_messages(user.uid, limit=15)
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for m in history:
        role = m.get("role", "user")
        messages.append({"role": role, "content": m.get("content", "")})

    # call OpenAI (response should be strict JSON per SYSTEM_MSG)
    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages + [{"role": "user", "content": req.message}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        log.exception("OpenAI error")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    parsed = parse_actions(text)
    reply_text = parsed.reply or "OK."

    # Apply actions safely
    # Nodes
    for n in parsed.actions.get("nodes", []):
        try:
            payload = UpsertNodeRequest(
                node_type=NodeType(n["node_type"]),
                quality=Quality(n["quality"]),
                energy=int(n.get("energy", 3)),
            )
            await upsert_node(payload, user)
        except Exception as e:
            log.warning(f"Node apply failed: {e}")

    # Connections
    for c in parsed.actions.get("connections", []):
        try:
            payload = UpsertConnRequest(
                source_node_id=c["source_node_id"],
                target_node_id=c["target_node_id"],
                action=Action(c["action"]),
                weight=float(c.get("weight", 1.0)),
                context=c.get("context", ""),
            )
            await upsert_connection(payload, user)
        except Exception as e:
            log.warning(f"Conn apply failed: {e}")

    # persist assistant msg
    persist_message(user.uid, "assistant", reply_text)

    return ChatResponse(reply=reply_text, actions=parsed.actions)

# -----------------------------
# Run (for local dev)
# -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

