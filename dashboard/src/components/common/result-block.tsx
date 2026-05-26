/** Shared JSON-result block — used by every admin mutation surface
 *  and by the memory inspector for "raw" payload reveal. Renders as
 *  a console transcript: a `>` prompt-rule above the JSON, hairline
 *  phosphor border on a sunken substrate. */
export function ResultBlock({ value }: { value: unknown }) {
  if (value == null) return null;
  return (
    <div className="mt-4 border border-border bg-surface-canvas font-mono text-xs">
      <div className="flex items-center gap-2 border-b border-border bg-muted/30 px-3 py-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
        <span className="text-primary">›</span>
        <span>response</span>
      </div>
      <pre className="max-h-56 overflow-auto p-3 leading-relaxed text-foreground/90">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}
