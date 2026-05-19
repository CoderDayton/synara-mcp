import { lazy } from "react";
import {
  createBrowserRouter,
  Navigate,
  useRouteError,
  isRouteErrorResponse,
} from "react-router-dom";
import { AppShell } from "@/components/layout/app-shell";
import Overview from "@/pages/overview";
import Memories from "@/pages/memories";
import Admin from "@/pages/admin";
import Config from "@/pages/config";

// Graph pulls in cytoscape + fcose (heavy) — split it out of the main
// chunk; the route element keeps it warm via <Activity> once visited.
const Graph = lazy(() => import("@/pages/graph"));

function RouteError() {
  const err = useRouteError();
  const msg = isRouteErrorResponse(err)
    ? `${err.status} ${err.statusText}`
    : err instanceof Error
      ? err.message
      : "Unexpected error";
  return (
    <div className="grid min-h-svh place-items-center bg-background px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-sm rounded-lg border border-border bg-card p-5 text-center sm:max-w-md sm:p-6">
        <h1 className="text-base font-semibold text-destructive sm:text-lg">
          Something went wrong
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">{msg}</p>
      </div>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Overview /> },
      { path: "memories", element: <Memories /> },
      { path: "graph", element: <Graph /> },
      { path: "admin", element: <Admin /> },
      { path: "config", element: <Config /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);
