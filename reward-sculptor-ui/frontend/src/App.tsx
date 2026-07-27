import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import { Icon } from "@/components/rs/icon";
import Layout from "@/components/Layout";
import Dashboard from "@/pages/Dashboard";
import LibraryPage from "@/pages/LibraryPage";
import ProjectCreate from "@/pages/ProjectCreate";
import ProjectDetail from "@/pages/ProjectDetail";
import ProjectList from "@/pages/ProjectList";
import Settings from "@/pages/Settings";

// Lazy — the saved viewer pulls in react-markdown; keep it out of the
// initial bundle like the other markdown-heavy route (Results tab).
const SavedMissionsPage = lazy(() => import("@/pages/SavedMissionsPage"));

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/projects" element={<ProjectList />} />
        <Route path="/projects/new" element={<ProjectCreate />} />
        <Route path="/projects/:slug" element={<ProjectDetail />} />
        <Route
          path="/saved"
          element={
            <Suspense fallback={<RouteFallback />}>
              <SavedMissionsPage />
            </Suspense>
          }
        />
        <Route path="/settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

function RouteFallback() {
  return (
    <div className="rs-scroll">
      <div className="rs-empty">
        <Icon name="loader" size={20} className="rs-spin" />
        <span className="rs-sub">Loading…</span>
      </div>
    </div>
  );
}

function NotFound() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="text-center">
        <h1 className="text-3xl font-semibold">404</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Page not found.
        </p>
      </div>
    </div>
  );
}
