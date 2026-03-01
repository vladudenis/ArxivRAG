"use client";

import { useState, useCallback, useRef, useEffect } from "react";

interface TopicsInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  id?: string;
  "aria-label"?: string;
}

const buildValue = (committed: string[], current: string): string => {
  const trimmed = current.trim();
  if (trimmed) {
    return [...committed, trimmed].join(", ");
  }
  return committed.length > 0 ? committed.join(", ") + "," : "";
};

const parseCommitted = (s: string): string[] => {
  const lastComma = s.lastIndexOf(",");
  if (lastComma === -1) return [];
  return s
    .substring(0, lastComma)
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
};

export const TopicsInput = ({
  value,
  onChange,
  placeholder = "Topics (comma to commit each term, spaces allowed)",
  disabled = false,
  id,
  "aria-label": ariaLabel = "Topics and tags for arXiv search",
}: TopicsInputProps) => {
  const [committedTerms, setCommittedTerms] = useState<string[]>(() =>
    parseCommitted(value)
  );
  const [inputKey, setInputKey] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const focusAfterCommitRef = useRef(false);

  useEffect(() => {
    if (focusAfterCommitRef.current) {
      focusAfterCommitRef.current = false;
      inputRef.current?.focus();
    }
  }, [inputKey]);

  const syncToParent = useCallback(
    (committed: string[], current: string) => {
      onChange(buildValue(committed, current));
    },
    [onChange]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      if (raw.includes(",")) {
        const parts = raw.split(",");
        const lastPart = parts.pop() ?? "";
        const newTerms = committedTerms.concat(
          parts.map((p) => p.trim()).filter(Boolean)
        );
        setCommittedTerms(newTerms);
        if (inputRef.current) {
          inputRef.current.value = lastPart;
        }
        if (!lastPart.trim()) {
          focusAfterCommitRef.current = true;
          setInputKey((k) => k + 1);
        }
        syncToParent(newTerms, lastPart);
      } else {
        syncToParent(committedTerms, raw);
      }
    },
    [committedTerms, syncToParent]
  );

  const handleRemoveTerm = useCallback(
    (index: number) => {
      const next = committedTerms.filter((_, i) => i !== index);
      setCommittedTerms(next);
      syncToParent(next, inputRef.current?.value ?? "");
    },
    [committedTerms, syncToParent]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      const currentVal = inputRef.current?.value ?? "";
      if (
        e.key === "Backspace" &&
        !currentVal &&
        committedTerms.length > 0
      ) {
        const next = committedTerms.slice(0, -1);
        const last = committedTerms[committedTerms.length - 1];
        setCommittedTerms(next);
        setInputKey((k) => k + 1);
        if (inputRef.current) {
          inputRef.current.value = last;
        }
        syncToParent(next, last);
        e.preventDefault();
      }
    },
    [committedTerms, syncToParent]
  );

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded-xl border border-zinc-300 bg-zinc-50 px-4 py-2 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent"
      onClick={() => inputRef.current?.focus()}
    >
      {committedTerms.map((term, i) => (
        <span
          key={`${term}-${i}`}
          className="inline-flex items-center gap-1 rounded-md bg-blue-100 px-2 py-1 text-sm text-blue-800"
        >
          {term}
          {!disabled && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleRemoveTerm(i);
              }}
              className="rounded p-0.5 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
              aria-label={`Remove ${term}`}
              tabIndex={0}
            >
              <span aria-hidden>×</span>
            </button>
          )}
        </span>
      ))}
      <input
        ref={inputRef}
        key={inputKey}
        id={id}
        type="text"
        defaultValue=""
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={committedTerms.length === 0 ? placeholder : "Add another..."}
        disabled={disabled}
        className="min-w-[120px] flex-1 bg-transparent py-1 text-sm text-zinc-900 placeholder-zinc-500 focus:outline-none"
        aria-label={ariaLabel}
      />
    </div>
  );
};
