import { useState } from "react";
import { Eye, Search } from "lucide-react";
import { useMemories } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Empty, ErrorState, Loading } from "@/components/common/states";
import { DeleteMemoryButton } from "@/components/memories/delete-memory-button";
import { MemoryDetailDialog } from "@/components/memories/memory-detail-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const PAGE = 25;
type Kind = "episodic" | "semantic";

export default function Memories() {
  const [kind, setKind] = useState<Kind>("episodic");
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [offset, setOffset] = useState(0);
  const [detailId, setDetailId] = useState<number | null>(null);

  const { data, isLoading, error, isFetching } = useMemories({
    kind,
    q: submitted || undefined,
    limit: PAGE,
    offset,
  });

  const rows =
    data && "items" in data
      ? (data.items as Array<{
          id?: number;
          content?: string;
          score?: number;
          metadata?: Record<string, unknown>;
        }>)
      : [];
  const isSearch = !!submitted;

  function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setOffset(0);
    setSubmitted(query.trim());
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Memories"
        subtitle="Browse, inspect, and prune episodic and semantic stores."
        actions={
          <form onSubmit={runSearch} className="flex w-full gap-2 sm:w-auto">
            <div className="relative flex-1 sm:w-64">
              <Search
                className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                aria-hidden
              />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={`Semantic search ${kind}…`}
                className="pl-8"
                aria-label="Search memories"
              />
            </div>
            <Button type="submit" variant="secondary">
              Search
            </Button>
            {isSearch && (
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  setQuery("");
                  setSubmitted("");
                }}
              >
                Clear
              </Button>
            )}
          </form>
        }
      />

      <Tabs
        value={kind}
        onValueChange={(v) => {
          setKind(v as Kind);
          setOffset(0);
          setSubmitted("");
          setQuery("");
        }}
      >
        <TabsList>
          <TabsTrigger value="episodic">Episodic</TabsTrigger>
          <TabsTrigger value="semantic">Semantic</TabsTrigger>
        </TabsList>
      </Tabs>

      {isLoading && <Loading />}
      {error && <ErrorState error={error} />}
      {!isLoading && !error && rows.length === 0 && (
        <Empty
          label={isSearch ? "No matches" : "Store is empty"}
          hint={
            isSearch
              ? "No memories scored against that query. Try broader terms or the other store."
              : `No ${kind} memories yet. Call store_episode from a connected MCP client to populate this store.`
          }
        />
      )}

      {rows.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-border/60 bg-card shadow-card [&_th]:text-[0.7rem] [&_th]:font-semibold [&_th]:uppercase [&_th]:tracking-wider [&_th]:text-muted-foreground [&_thead]:bg-muted/40">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-20">ID</TableHead>
                <TableHead>Content</TableHead>
                {isSearch && <TableHead className="w-24">Score</TableHead>}
                <TableHead className="w-28 text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => (
                <TableRow key={r.id ?? i}>
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {r.id != null ? `#${r.id}` : "—"}
                  </TableCell>
                  <TableCell className="max-w-md">
                    <span className="line-clamp-2">{r.content ?? "—"}</span>
                  </TableCell>
                  {isSearch && (
                    <TableCell className="font-mono text-xs tabular-nums">
                      {typeof r.score === "number" ? r.score.toFixed(3) : "—"}
                    </TableCell>
                  )}
                  <TableCell className="text-right">
                    {r.id != null && (
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label={`Inspect episode ${r.id}`}
                          onClick={() => setDetailId(r.id as number)}
                        >
                          <Eye className="size-4" aria-hidden />
                        </Button>
                        {kind === "episodic" && (
                          <DeleteMemoryButton id={r.id} />
                        )}
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {!isSearch && rows.length > 0 && (
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            Showing {offset + 1}–{offset + rows.length}
            {isFetching && " · updating…"}
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={rows.length < PAGE}
              onClick={() => setOffset(offset + PAGE)}
            >
              Next
            </Button>
          </div>
        </div>
      )}

      <MemoryDetailDialog
        id={detailId}
        onOpenChange={(o) => !o && setDetailId(null)}
      />
    </div>
  );
}
