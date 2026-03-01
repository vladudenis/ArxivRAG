import type { Source } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

const escapeHtml = (s: string): string =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const formatSource = (source: Source): string =>
  `<a href="${escapeHtml(source.arxiv_url)}" target="_blank" rel="noopener noreferrer" class="source-link">${escapeHtml(source.title)}${source.section && source.section !== "Abstract" ? ` <span class="source-section">(${escapeHtml(source.section)})</span>` : ""}</a>`;

export const generateSessionHtml = (messages: Message[]): string => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const title = `ArxivRAG Session ${timestamp}`;

  const items: string[] = messages.map((msg) => {
    const isUser = msg.role === "user";
    const bubbleClass = isUser ? "bubble-user" : "bubble-assistant";
    const alignClass = isUser ? "align-end" : "align-start";

    let sourcesHtml = "";
    if (!isUser && msg.sources && msg.sources.length > 0) {
      sourcesHtml = `
        <div class="sources-block">
          <p class="sources-label">Sources</p>
          <ul class="sources-list">
            ${msg.sources.map((s) => `<li>${formatSource(s)}</li>`).join("")}
          </ul>
        </div>`;
    }

    return `
      <div class="message ${alignClass}">
        <div class="${bubbleClass}">
          <p class="content">${escapeHtml(msg.content).replace(/\n/g, "<br>")}</p>
          ${sourcesHtml}
        </div>
      </div>`;
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)}</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #fafafa;
      color: #171717;
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }
    .header {
      max-width: 48rem;
      margin: 0 auto 24px;
      padding-bottom: 12px;
      border-bottom: 1px solid #e4e4e7;
    }
    .header h1 { font-size: 1.125rem; font-weight: 600; margin: 0 0 4px 0; }
    .header p { font-size: 0.75rem; color: #71717a; margin: 0; }
    .chat {
      max-width: 48rem;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .message { display: flex; width: 100%; }
    .align-end { justify-content: flex-end; }
    .align-start { justify-content: flex-start; }
    .bubble-user, .bubble-assistant {
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 1rem;
    }
    .bubble-user {
      background: #2563eb;
      color: white;
    }
    .bubble-assistant {
      background: #f4f4f5;
      color: #18181b;
    }
    .content {
      font-size: 0.875rem;
      margin: 0;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .sources-block {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #e4e4e7;
    }
    .sources-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #71717a;
      margin: 0 0 8px 0;
    }
    .sources-list {
      list-style: none;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .source-link {
      font-size: 0.875rem;
      color: #2563eb;
      text-decoration: none;
    }
    .source-link:hover { text-decoration: underline; }
    .source-section { color: #71717a; font-weight: normal; }
  </style>
</head>
<body>
  <header class="header">
    <h1>ArxivRAG</h1>
    <p>Session exported ${new Date().toLocaleString()}</p>
  </header>
  <main class="chat">
    ${items.join("")}
  </main>
</body>
</html>`;
};

export const downloadSessionHtml = (messages: Message[]): void => {
  const html = generateSessionHtml(messages);
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `arxivrag-session-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
