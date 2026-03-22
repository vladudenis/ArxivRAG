"use client";

import { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { SourcesList } from "./SourcesList";
import type { StrategyResult } from "@/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  results?: StrategyResult[];
}

export const ChatMessage = ({ role, content, results }: ChatMessageProps) => {
  const isUser = role === "user";
  const [currentPage, setCurrentPage] = useState(0);
  const hasPagination = results && results.length > 1;
  const totalPages = results?.length ?? 0;
  const currentResult = results?.[currentPage];

  const handlePrev = useCallback(() => {
    setCurrentPage((p) => Math.max(0, p - 1));
  }, []);

  const handleNext = useCallback(() => {
    setCurrentPage((p) => Math.min(totalPages - 1, p + 1));
  }, [totalPages]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, handler: () => void) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handler();
      }
    },
    [],
  );

  const displayContent = currentResult?.answer ?? content;
  const displaySources = currentResult?.sources;
  const strategyLabel = currentResult?.strategy_label;

  return (
    <div
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
      role="article"
      aria-label={`${role} message`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser ? "bg-blue-600 text-white" : "bg-zinc-100 text-zinc-900"
        }`}
      >
        {!isUser && hasPagination && (
          <div
            className="mb-3 flex items-center justify-between gap-4 rounded-lg bg-zinc-200/60 px-3 py-2 dark:bg-zinc-700/40"
            role="navigation"
            aria-label="Strategy pagination"
          >
            <span className="text-xs font-medium text-zinc-700 dark:text-white">
              {strategyLabel}
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={handlePrev}
                onKeyDown={(e) => handleKeyDown(e, handlePrev)}
                disabled={currentPage === 0}
                className="rounded p-1.5 text-zinc-600 hover:bg-zinc-300 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Previous strategy"
                tabIndex={0}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M15 18l-6-6 6-6" />
                </svg>
              </button>
              <span className="min-w-16 text-center text-xs text-zinc-600 dark:text-white">
                {currentPage + 1} / {totalPages}
              </span>
              <button
                type="button"
                onClick={handleNext}
                onKeyDown={(e) => handleKeyDown(e, handleNext)}
                disabled={currentPage >= totalPages - 1}
                className="rounded p-1.5 text-zinc-600 hover:bg-zinc-300 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Next strategy"
                tabIndex={0}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M9 18l6-6-6-6" />
                </svg>
              </button>
            </div>
          </div>
        )}
        <div className="text-sm leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:ml-4 [&_ol]:list-decimal [&_ol]:ml-4 [&_li]:my-0.5">
          <ReactMarkdown>{displayContent}</ReactMarkdown>
        </div>
        {!isUser && displaySources && displaySources.length > 0 && (
          <SourcesList sources={displaySources} className="mt-3" />
        )}
      </div>
    </div>
  );
};
