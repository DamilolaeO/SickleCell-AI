// Talks to the FastAPI chatbot (backend/app/main.py).
// The address can be changed with NEXT_PUBLIC_API_URL in frontend/.env.local.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

// These match the Source and Answer models in backend/app/main.py
export type Source = {
  title: string;
  owner: string;
  licence: string;
  url: string;
  kind: string; // "guidance" or "research"
};

export type ReplyType =
  | "answer"
  | "emergency"
  | "crisis_support"
  | "dose_refusal"
  | "dont_know";

export type Reply = {
  type: ReplyType;
  answer: string;
  sources: Source[];
};

export async function askQuestion(question: string): Promise<Reply> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    throw new Error(`Server replied ${res.status}`);
  }
  return res.json();
}
