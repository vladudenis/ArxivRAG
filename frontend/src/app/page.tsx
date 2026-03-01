"use client";

import { useState, useCallback } from "react";
import { ChatMessage } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { sendQuery, type Source } from "@/lib/api";
import { downloadSessionHtml } from "@/lib/sessionExport";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = useCallback(async (query: string, topics: string) => {
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await sendQuery(query, topics);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
        },
      ]);
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Failed to get response";
      setError(errMsg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${errMsg}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-white">
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-6">
        <header className="flex items-center justify-between gap-4 border-b border-zinc-200 py-3">
          <div>
            <h1 className="text-lg font-semibold text-zinc-900">
              ArxivRAG
            </h1>
            <p className="text-xs text-zinc-500">
              Ask questions about arXiv papers
            </p>
          </div>
          <button
            type="button"
            onClick={() => downloadSessionHtml(messages)}
            disabled={messages.length === 0 || loading}
            className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Save session as HTML"
          >
            Save session
          </button>
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="py-8 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <p className="text-zinc-500 mb-2">
                Enter topics for arXiv search and a question to get an AI-powered answer.
              </p>
              <p className="text-sm text-zinc-400">
                Example topics: Agentic RAG, transformer, hallucination
              </p>
              <p className="text-sm text-zinc-400 mt-1">
                Example question: &quot;How does RAG reduce hallucination?&quot;
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              role={msg.role}
              content={msg.content}
              sources={msg.sources}
            />
          ))}

          {loading && (
            <div className="flex justify-start" role="status" aria-label="Loading">
              <div className="max-w-[85%] rounded-2xl bg-zinc-100 px-4 py-3">
                <p className="text-sm text-zinc-500">
                  Searching arXiv, processing papers...
                </p>
              </div>
            </div>
          )}

          {error && (
            <p className="text-sm text-red-600" role="alert">
              {error}
            </p>
          )}
          </div>
        </main>

        <div className="sticky bottom-0">
          <ChatInput onSend={handleSend} disabled={loading} />
        </div>
      </div>
    </div>
  );
}
