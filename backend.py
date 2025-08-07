import uuid
from datetime import datetime
from typing import Optional
import os
import base64
import tempfile

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import firebase_admin
from firebase_admin import credentials, auth, firestore

import stripe
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Configs ---
# Stripe keys from environment
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://aquarian-8c213.web.app")

if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET:
    raise RuntimeError("Stripe keys must be set in environment variables.")

# Firebase key: support both local file and Base64 environment variable
firebase_key_b64 = os.environ.get("FIREBASE_KEY_B64")
if firebase_key_b64:
    key_data = base64.b64decode(firebase_key_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(key_data)
        FIREBASE_KEY_PATH = f.name
else:
    FIREBASE_KEY_PATH = "config/aquarian_key.json"

# --- Firebase Init ---
cred = credentials.Certificate(FIREBASE_KEY_PATH)
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Stripe Init ---
stripe.api_key = STRIPE_SECRET_KEY

# --- FastAPI App ---
app = FastAPI(title="Aquarian")

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        FRONTEND_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth Helper ---
bearer_scheme = HTTPBearer()

class User(BaseModel):
    uid: str
    email: str
    pro: bool = False

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not token or not token.credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token.")
    try:
        decoded_token = auth.verify_id_token(token.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    user_ref = db.collection('users').document(decoded_token['uid'])
    user_doc = user_ref.get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    return User(
        uid=decoded_token['uid'],
        email=decoded_token.get('email', ''),
        pro=user_data.get('pro', False)
    )

# --- Pydantic Models ---
class Node(BaseModel):
    node_id: str
    node_type: str
    quality: str
    energy: int
    last_updated: str

class Connection(BaseModel):
    source_node_id: str
    target_node_id: str
    action: str
    weight: float
    effect: Optional[float] = None
    context: Optional[str] = ""

class UpsertNodeRequest(BaseModel):
    node_type: str
    quality: str
    energy: int

class UpsertConnRequest(BaseModel):
    source_node_id: str
    target_node_id: str
    action: str
    weight: float
    context: Optional[str] = ""

class CreatePaymentIntentRequest(BaseModel):
    uid: str

# --- Endpoints ---
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}

@app.post("/user/me", tags=["User"])
async def create_or_get_user(user=Depends(get_current_user)):
    user_ref = db.collection('users').document(user.uid)
    if not user_ref.get().exists:
        user_ref.set({"email": user.email, "pro": False})
    return {"uid": user.uid, "email": user.email, "pro": user.pro}

@app.post("/network/node", tags=["Network"])
async def upsert_node(req: UpsertNodeRequest, user=Depends(get_current_user)):
    existing = db.collection("users").document(user.uid).collection("nodes") \
        .where("node_type", "==", req.node_type) \
        .where("quality", "==", req.quality) \
        .limit(1).stream()

    now = datetime.utcnow().isoformat()

    if existing:
        for doc in existing:
            doc.reference.update({
                "energy": max(1, min(5, req.energy)),
                "last_updated": now
            })
            return {**doc.to_dict(), "energy": req.energy, "last_updated": now}

    node_id = f"N{uuid.uuid4().hex[:8]}"
    node = {
        "node_id": node_id,
        "node_type": req.node_type,
        "quality": req.quality,
        "energy": max(1, min(5, req.energy)),
        "last_updated": now,
    }
    db.collection("users").document(user.uid).collection("nodes").document(node_id).set(node)
    return node

@app.get("/network", tags=["Network"])
async def fetch_network(user=Depends(get_current_user)):
    nodes = [n.to_dict() for n in db.collection("users").document(user.uid).collection("nodes").stream()]
    conns = [c.to_dict() for c in db.collection("users").document(user.uid).collection("connections").stream()]
    return {"nodes": nodes, "connections": conns}

@app.post("/network/connection", tags=["Network"])
async def upsert_connection(req: UpsertConnRequest, user=Depends(get_current_user)):
    conn_id = f"C{uuid.uuid4().hex[:8]}"
    conn = {
        "source_node_id": req.source_node_id,
        "target_node_id": req.target_node_id,
        "action": req.action,
        "weight": req.weight,
        "effect": None,
        "context": req.context,
        "created": datetime.utcnow().isoformat(),
    }
    db.collection("users").document(user.uid).collection("connections").document(conn_id).set(conn)
    return conn

@app.get("/pro/status", tags=["Pro"])
async def check_pro(user=Depends(get_current_user)):
    if not user.pro:
        raise HTTPException(status_code=402, detail="Pro required")
    return {"pro": True}

@app.post("/create-payment-intent", tags=["Stripe"])
async def create_payment_intent(req: CreatePaymentIntentRequest):
    try:
        intent = stripe.PaymentIntent.create(
            amount=500,  # $5.00 in cents
            currency="usd",
            payment_method_types=["card"],
            metadata={"uid": req.uid},
            description="Aquarian Pro Subscription"
        )
        return {"clientSecret": intent.client_secret}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stripe/webhook", tags=["Stripe"])
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Webhook verification failed")

    if event['type'] == 'payment_intent.succeeded':
        intent = event['data']['object']
        uid = intent['metadata'].get('uid')
        if uid:
            user_ref = db.collection('users').document(uid)
            if user_ref.get().exists:
                user_ref.update({'pro': True})

    return {"status": "success"}
