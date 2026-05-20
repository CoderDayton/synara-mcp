/** Shared JSON-result block — used by every admin mutation surface
 *  and by the memory inspector for "raw" payload reveal. */
export function ResultBlock({ value }: { value: unknown }) {
  if (value == null) return null;
  return (
    <pre className="mt-4 max-h-56 overflow-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground/90">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
