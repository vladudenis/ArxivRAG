"use client";

import type { Source } from "@/lib/api";

interface SourcesListProps {
  sources: Source[];
  className?: string;
}

export const SourcesList = ({ sources, className = "" }: SourcesListProps) => {
  if (sources.length === 0) return null;

  return (
    <div className={`border-t border-zinc-200 pt-3 ${className}`}>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
        Sources
      </p>
      <ul className="space-y-2" role="list">
        {sources.map((source) => (
          <li key={source.paper_id}>
            <a
              href={source.arxiv_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
              aria-label={`Open ${source.title} on arXiv`}
            >
              {source.title}
              {source.section && source.section !== "Abstract" && (
                <span className="ml-1 text-zinc-500">
                  ({source.section})
                </span>
              )}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
};
