"use client";

import { SourcesList } from "./SourcesList";
import type { Source } from "@/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export const ChatMessage = ({
  role,
  content,
  sources,
}: ChatMessageProps) => {
  const isUser = role === "user";

  return (
    <div
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
      role="article"
      aria-label={`${role} message`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-zinc-100 text-zinc-900"
        }`}
      >
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{content}</p>
        {!isUser && sources && sources.length > 0 && (
          <SourcesList sources={sources} className="mt-3" />
        )}
      </div>
    </div>
  );
};
