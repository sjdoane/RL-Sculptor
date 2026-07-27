import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null };

/** Last-resort recovery for an unexpected render failure.
 *
 * Normal request failures stay local to their dialog/card. This boundary only
 * prevents one malformed response or third-party component from leaving the
 * entire research workspace as an unrecoverable white page.
 */
export class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("RL Sculptor interface error", error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <main
        role="alert"
        style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          padding: 24,
          background: "var(--canvas)",
          color: "var(--ink)",
        }}
      >
        <section className="rs-card rs-card-pad" style={{ width: "min(520px, 100%)" }}>
          <p className="rs-kicker" style={{ marginTop: 0 }}>Interface recovery</p>
          <h1 className="rs-h2" style={{ marginBottom: 8 }}>RL Sculptor hit an interface error</h1>
          <p className="rs-sub" style={{ marginBottom: 20 }}>
            Your project and run data are still on disk. Reload the workspace to restore the latest state.
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <button className="rs-btn rs-btn-primary" onClick={() => window.location.reload()}>
              Reload workspace
            </button>
            <a className="rs-btn rs-btn-ghost" href="/">Return to dashboard</a>
          </div>
        </section>
      </main>
    );
  }
}
