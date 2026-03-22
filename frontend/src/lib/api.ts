export interface Source {
  paper_id: string;
  title: string;
  section: string;
  arxiv_url: string;
}

export interface StrategyResult {
  strategy: string;
  strategy_label: string;
  answer: string;
  sources: Source[];
}

export interface QueryResponse {
  results: StrategyResult[];
}

const getApiUrl = (): string => {
  if (typeof window !== "undefined") {
    return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  }
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
};

export const sendQuery = async (
  query: string,
  topics: string
): Promise<QueryResponse> => {
  const url = getApiUrl();
  const res = await fetch(`${url}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, topics }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Query failed");
  }

  return res.json();
};
