"use client";

import ReactMarkdown from "react-markdown";
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
        <div className="text-sm leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:ml-4 [&_ol]:list-decimal [&_ol]:ml-4 [&_li]:my-0.5">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
        {!isUser && sources && sources.length > 0 && (
          <SourcesList sources={sources} className="mt-3" />
        )}
      </div>
    </div>
  );
};
