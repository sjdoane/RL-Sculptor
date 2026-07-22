import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";

import App from "@/App";
import { AppErrorBoundary } from "@/components/AppErrorBoundary";
import { bootstrapTheme } from "@/hooks/useTheme";
import "@/index.css";
import "@/styles/rs-tokens.css";
import "@/styles/rs-theme.css";

// Apply the stored theme (.dark class + data-theme attr) before React
// mounts so the first paint is correct (no flash of the wrong theme).
bootstrapTheme();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        // Don't retry 4xx — those are actionable errors, not transient.
        const status = (error as { status?: number })?.status;
        if (status && status >= 400 && status < 500) return false;
        return failureCount < 2;
      },
      staleTime: 30_000,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppErrorBoundary>
          <App />
          <Toaster position="bottom-right" richColors closeButton />
        </AppErrorBoundary>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
