"use client";

import { useState, useCallback } from "react";
import { TopicsInput } from "@/components/TopicsInput";

interface ChatInputProps {
  onSend: (query: string, topics: string) => void;
  disabled?: boolean;
}

const hasValidTopics = (topics: string): boolean => {
  const trimmed = topics.trim();
  if (!trimmed) return false;
  const terms = trimmed
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  return terms.length >= 1;
};

export const ChatInput = ({ onSend, disabled = false }: ChatInputProps) => {
  const [query, setQuery] = useState("");
  const [topics, setTopics] = useState("");
  const [topicsResetKey, setTopicsResetKey] = useState(0);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const trimmedQuery = query.trim();
      const trimmedTopics = topics.trim();
      if (
        trimmedQuery &&
        hasValidTopics(topics) &&
        !disabled
      ) {
        onSend(trimmedQuery, trimmedTopics);
        setQuery("");
        setTopics("");
        setTopicsResetKey((k) => k + 1);
      }
    },
    [query, topics, onSend, disabled]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, field: "query" | "topics") => {
      if (e.key === "Enter" && !e.shiftKey && field === "query") {
        e.preventDefault();
        handleSubmit(e as unknown as React.FormEvent);
      }
    },
    [handleSubmit]
  );

  const canSubmit =
    query.trim().length > 0 &&
    hasValidTopics(topics) &&
    !disabled;

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 p-6 bg-white border-t border-zinc-200"
    >
      <div className="flex flex-col gap-1">
        <label
          htmlFor="topics-input"
          className="text-xs font-medium text-zinc-600"
        >
          Topics / tags <span className="text-red-500">*</span>
        </label>
        <TopicsInput
          key={topicsResetKey}
          id="topics-input"
          value={topics}
          onChange={setTopics}
          disabled={disabled}
          aria-label="Topics and tags for arXiv search (comma to lock each term)"
        />
      </div>
      <div className="flex gap-2">
        <input
          id="query-input"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => handleKeyDown(e, "query")}
          placeholder="Ask about arXiv papers..."
          disabled={disabled}
          className="flex-1 rounded-xl border border-zinc-300 bg-zinc-50 px-4 py-3 text-sm text-zinc-900 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Message input"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-600"
          aria-label="Send message"
        >
          Send
        </button>
      </div>
    </form>
  );
};
