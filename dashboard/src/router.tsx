import { lazy } from "react";
import {
  createBrowserRouter,
  Navigate,
  useRouteError,
  isRouteErrorResponse,
} from "react-router-dom";
import { AppShell } from "@/components/layout/app-shell";

// Route-level code splitting: each page becomes its own chunk so the
// initial bundle ships only the overview path. AppShell's <Suspense>
// covers the loading state via the same `<Loading/>` fallback used
// elsewhere.
const Overview = lazy(() => import("@/pages/overview"));
const Memories = lazy(() => import("@/pages/memories"));
const Admin = lazy(() => import("@/pages/admin"));
const Config = lazy(() => import("@/pages/config"));

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
      { path: "admin", element: <Admin /> },
      { path: "config", element: <Config /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);
