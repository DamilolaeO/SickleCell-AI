"use client";

import { useEffect, useRef, useState } from "react";
import { askQuestion, type Reply, type Source } from "@/lib/api";
import { Button } from "@/components/ui/button";

type Message =
  | { role: "user"; text: string }
  | { role: "bot"; reply: Reply }
  | { role: "error" };

// The fixed NHS urgent note that ask.py adds to the end of some answers
const URGENT_NOTE_START = "IMPORTANT: The NHS says";

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll to the newest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (q.length < 2 || loading) return;

    setMessages((m) => [...m, { role: "user", text: q }]);
    setQuestion("");
    setLoading(true);
    try {
      const reply = await askQuestion(q);
      setMessages((m) => [...m, { role: "bot", reply }]);
    } catch {
      setMessages((m) => [...m, { role: "error" }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-1 flex-col">
      <div className="flex-1 space-y-4 py-4">
        {messages.length === 0 && (
          <p className="text-muted-foreground">
            Ask a question about sickle cell disease. For example: &quot;What is a sickle cell
            crisis?&quot; or &quot;Why does sickle cell cause tiredness?&quot;
          </p>
        )}

        {messages.map((m, i) => {
          if (m.role === "user") return <UserBubble key={i} text={m.text} />;
          if (m.role === "error") return <ErrorBox key={i} />;
          return <BotReply key={i} reply={m.reply} />;
        })}

        {loading && <p className="text-muted-foreground">Looking for trusted information...</p>}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={send} className="sticky bottom-0 flex gap-2 border-t bg-background py-4">
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <input
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          maxLength={500}
          placeholder="Type your question"
          className="h-10 flex-1 rounded-lg border border-input bg-background px-3 outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        />
        <Button type="submit" size="lg" className="h-10 px-4" disabled={loading || question.trim().length < 2}>
          Ask
        </Button>
      </form>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[80%] rounded-2xl bg-primary px-4 py-2 text-primary-foreground">{text}</p>
    </div>
  );
}

function BotReply({ reply }: { reply: Reply }) {
  switch (reply.type) {
    case "emergency":
      return <EmergencyAlert text={reply.answer} />;
    case "dose_refusal":
    case "dont_know":
      return <Signpost text={reply.answer} />;
    default:
      return <Answer text={reply.answer} sources={reply.sources} />;
  }
}

// Emergency: big red alert with 999 and 111 buttons
function EmergencyAlert({ text }: { text: string }) {
  return (
    <div role="alert" className="rounded-xl border-2 border-red-700 bg-red-50 p-5 text-red-950">
      <p className="text-xl font-bold">Call 999 now if this is an emergency</p>
      <p className="mt-3 whitespace-pre-line">{text}</p>
      <div className="mt-4 flex flex-wrap gap-3">
        <a href="tel:999" className="rounded-lg bg-red-700 px-5 py-3 text-lg font-bold text-white hover:bg-red-800">
          Call 999
        </a>
        <a href="tel:111" className="rounded-lg border-2 border-red-700 px-5 py-3 text-lg font-bold text-red-800 hover:bg-red-100">
          Call NHS 111
        </a>
        <a href="https://111.nhs.uk/" target="_blank" rel="noopener noreferrer" className="self-center font-medium text-red-800 underline">
          NHS 111 online
        </a>
      </div>
    </div>
  );
}

// Dose refusal and "I don't know": calm box that points to people who can help
function Signpost({ text }: { text: string }) {
  return (
    <div className="rounded-xl border border-blue-300 bg-blue-50 p-4 text-blue-950">
      <p className="whitespace-pre-line">
        <Linkify text={text} />
      </p>
      <p className="mt-3 font-semibold">Where to get help</p>
      <ul className="mt-1 list-disc space-y-1 pl-5">
        <li>Your haematology team or GP</li>
        <li>A pharmacist, for questions about medicines</li>
        <li>
          <a href="https://111.nhs.uk/" target="_blank" rel="noopener noreferrer" className="underline">
            NHS 111 online
          </a>{" "}
          or call 111
        </li>
        <li>In an emergency, call 999</li>
      </ul>
    </div>
  );
}

// Normal answer: text with [1] style citations, an optional NHS urgent note, and numbered sources
function Answer({ text, sources }: { text: string; sources: Source[] }) {
  const cut = text.indexOf(URGENT_NOTE_START);
  const body = cut >= 0 ? text.slice(0, cut).trim() : text;
  const urgent = cut >= 0 ? text.slice(cut).trim() : null;

  return (
    <div className="rounded-xl border bg-card p-4">
      <p className="whitespace-pre-line">
        <Citations text={body} count={sources.length} />
      </p>

      {urgent && (
        <div className="mt-4 rounded-lg border border-amber-400 bg-amber-50 p-3 text-amber-950">
          <p className="whitespace-pre-line">{urgent}</p>
        </div>
      )}

      {sources.length > 0 && (
        <div className="mt-4 border-t pt-3">
          <p className="text-sm font-semibold">Sources</p>
          <ol className="mt-1 list-decimal space-y-1 pl-5 text-sm">
            {sources.map((s, i) => (
              <li key={s.url} id={`source-${i + 1}`}>
                <SafeLink url={s.url}>{s.title}</SafeLink>
                <span className="text-muted-foreground">
                  {" "}
                  ({s.owner}
                  {s.kind === "research" ? ", research paper" : ""})
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

// Turns [1], [2] in the answer into small links that jump to that source
function Citations({ text, count }: { text: string; count: number }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const n = part.match(/^\[(\d+)\]$/)?.[1];
        if (n && Number(n) >= 1 && Number(n) <= count) {
          return (
            <a key={i} href={`#source-${n}`} className="align-super text-xs font-semibold text-primary underline">
              [{n}]
            </a>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// Makes web addresses in fixed messages clickable
function Linkify({ text }: { text: string }) {
  const parts = text.split(/(https?:\/\/[^\s]+)/g);
  return (
    <>
      {parts.map((part, i) =>
        /^https?:\/\//.test(part) ? (
          <SafeLink key={i} url={part}>
            {part}
          </SafeLink>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

// Only opens normal web links (never javascript: or other odd addresses)
function SafeLink({ url, children }: { url: string; children: React.ReactNode }) {
  if (!/^https?:\/\//i.test(url)) return <>{children}</>;
  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className="text-primary underline">
      {children}
    </a>
  );
}

function ErrorBox() {
  return (
    <div role="alert" className="rounded-xl border border-red-300 bg-red-50 p-4 text-red-950">
      <p className="font-semibold">Sorry, the assistant is not working right now.</p>
      <p className="mt-1">
        Please try again in a moment. If you need help now, call NHS 111. In an emergency, call 999.
      </p>
    </div>
  );
}
