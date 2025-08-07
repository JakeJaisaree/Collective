import os
import subprocess
import platform
import shutil

# Create backend and frontend folders
os.makedirs("backend/config", exist_ok=True)
os.makedirs("frontend/src/components", exist_ok=True)

# Move aquarian_key.json to backend/config
if os.path.exists("aquarian_key.json") and not os.path.exists("backend/config/aquarian_key.json"):
    shutil.move("aquarian_key.json", "backend/config/aquarian_key.json")
    print("Moved aquarian_key.json to backend/config/")

# Move Backend.py to backend/main.py
if os.path.exists("Backend.py") and not os.path.exists("backend/main.py"):
    shutil.move("Backend.py", "backend/main.py")
    print("Moved Backend.py to backend/main.py")

# Create frontend App.jsx
app_path = "frontend/src/App.jsx"
if not os.path.exists(app_path):
    with open(app_path, "w") as f:
        f.write("""\
import ProCheckout from "./components/ProCheckout";

function App() {
  return (
    <div style={{ padding: "2rem" }}>
      <h1>Aquarian</h1>
      <ProCheckout />
    </div>
  );
}

export default App;
""")

# Create frontend ProCheckout.js
checkout_path = "frontend/src/components/ProCheckout.js"
if not os.path.exists(checkout_path):
    with open(checkout_path, "w") as f:
        f.write("""\
import { useState, useEffect } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { Elements, PaymentElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { getAuth } from "firebase/auth";

const stripePromise = loadStripe("pk_test_your_key_here"); // Replace with your key

function CheckoutForm({ uid }) {
  const stripe = useStripe();
  const elements = useElements();
  const [clientSecret, setClientSecret] = useState("");

  useEffect(() => {
    getAuth().currentUser?.getIdToken().then(token => {
      fetch("/api/create-payment-intent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({ uid }),
      })
      .then(res => res.json())
      .then(data => setClientSecret(data.clientSecret));
    });
  }, [uid]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const { error } = await stripe.confirmPayment({ elements });
    if (!error) alert("Upgrade successful!");
  };

  if (!clientSecret) return <div>Loading payment form...</div>;
  return (
    <form onSubmit={handleSubmit}>
      <PaymentElement />
      <button disabled={!stripe}>Pay</button>
    </form>
  );
}

export default function ProCheckout() {
  const [uid, setUid] = useState(null);
  useEffect(() => {
    const unsub = getAuth().onAuthStateChanged(user => setUid(user?.uid ?? null));
    return () => unsub();
  }, []);
  if (!uid) return <div>Sign in to upgrade</div>;
  return (
    <Elements stripe={stripePromise} options={{ clientSecret: "your-client-secret" }}>
      <CheckoutForm uid={uid} />
    </Elements>
  );
}
""")

# Create batch script to setup frontend
bat_script = "setup-frontend.bat"
if not os.path.exists(bat_script):
    with open(bat_script, "w") as f:
        f.write("""@echo off
echo Creating Vite React app...
npm create vite@latest frontend -- --template react
cd frontend
call npm install
call npm install firebase @stripe/react-stripe-js @stripe/stripe-js
echo React frontend and dependencies installed!
""")

# Create unified start-all.bat script
with open("start-all.bat", "w") as f:
    f.write("""@echo off
start cmd /k "cd backend && uvicorn main:app --reload"
start cmd /k "cd frontend && npm run dev"
""")

print("\nProject structure initialized.")

# Ask user to run frontend setup
run_now = input("Do you want to run the frontend setup now? (y/n): ").strip().lower()
if run_now == "y":
    if platform.system() == "Windows":
        subprocess.run(["setup-frontend.bat"], shell=True)
    else:
        print("Windows-only automation. On macOS/Linux, run these manually:\n  npm create vite@latest frontend -- --template react\n  cd frontend\n  npm install")