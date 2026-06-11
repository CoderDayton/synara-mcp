/**
 * Episode/schema content renderer.
 *
 * Agents store either markdown-ish notes (headings, lists, fences) or
 * dense plain text. A deterministic structure sniff picks the right
 * presentation; a raw/format toggle lets the user override it, because
 * no heuristic survives every LLM's formatting habits. Markdown renders
 * through react-markdown + GFM with raw HTML disabled (untrusted input
 * stays inert); plain text keeps the classic pre-wrap block.
 */
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Compact dark-theme markdown styling, scoped via the components map
 *  (no global typography plugin). Sizes mirror the plain-text block so
 *  toggling modes doesn't jump the layout. */
const MD_COMPONENTS: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  h1: ({ children }) => (
    <h1 className="mt-3 mb-1.5 text-base font-semibold text-foreground first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-3 mb-1.5 text-sm font-semibold text-foreground first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-2.5 mb-1 text-[0.8rem] font-semibold text-foreground first:mt-0">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-2.5 mb-1 text-[0.8rem] font-semibold text-foreground/90 first:mt-0">
      {children}
    </h4>
  ),
  p: ({ children }) => <p className="my-1.5 leading-relaxed first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => (
    <ul className="my-1.5 list-disc space-y-0.5 pl-5 marker:text-muted-foreground/60">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-1.5 list-decimal space-y-0.5 pl-5 marker:text-muted-foreground/60">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-primary underline decoration-primary/40 underline-offset-2 hover:decoration-primary"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground italic">
      {children}
    </blockquote>
  ),
  code: ({ children, className }) =>
    className ? (
      // Block code (inside <pre>): the fence's language lands in
      // className; let the pre wrapper style it.
      <code className="font-mono">{children}</code>
    ) : (
      <code className="rounded bg-foreground/10 px-1 py-0.5 font-mono text-[0.75rem]">
        {children}
      </code>
    ),
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-md border border-border bg-background/60 p-2.5 font-mono text-[0.75rem] leading-relaxed">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-[0.75rem]">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-foreground/5 px-2 py-1 text-left font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
  hr: () => <hr className="my-3 border-border" />,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
};

/**
 * Render memory content with a formatted/raw toggle. Formatted
 * (markdown) is the default — agents overwhelmingly write
 * markdown-shaped notes, and plain prose renders as paragraphs
 * anyway; the verbatim text stays one click away.
 */
export function MemoryContent({ text }: { text: string }) {
  const [formatted, setFormatted] = useState(true);

  return (
    <div className="relative">
      <div className="absolute top-1.5 right-1.5 z-10 flex overflow-hidden rounded border border-border/70 bg-surface-overlay/90 font-mono text-[0.58rem] backdrop-blur sm:text-[0.6rem]">
        {(["formatted", "raw"] as const).map((m) => {
          const active = formatted === (m === "formatted");
          return (
            <button
              key={m}
              type="button"
              aria-pressed={active}
              title={m === "formatted" ? "Render markdown structure" : "Show the stored text verbatim"}
              onClick={() => setFormatted(m === "formatted")}
              className={`px-1.5 py-0.5 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary ${
                active
                  ? "bg-foreground/10 text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m}
            </button>
          );
        })}
      </div>
      {formatted ? (
        <div className="rounded-md border border-border bg-muted/40 p-3 pt-7 font-sans text-[0.85rem] text-foreground/90">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
            {text}
          </ReactMarkdown>
        </div>
      ) : (
        <pre className="rounded-md border border-border bg-muted/40 p-3 pt-7 text-xs leading-relaxed break-words whitespace-pre-wrap text-foreground">
          {text}
        </pre>
      )}
    </div>
  );
}
