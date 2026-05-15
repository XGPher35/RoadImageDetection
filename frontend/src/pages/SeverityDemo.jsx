import { Link } from 'react-router-dom';

export default function SeverityDemo() {
  return (
    <div className="min-h-screen bg-base text-text-primary">
      <header className="border-b border-border">
        <div className="site-container h-14 flex items-center justify-between">
          <Link to="/" className="font-heading font-semibold text-sm tracking-tight">
            SHP
          </Link>
          <Link
            to="/"
            className="text-sm text-text-muted hover:text-text-primary transition-colors flex items-center gap-1.5"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
            Back
          </Link>
        </div>
      </header>

      <div className="site-container py-12">
        <h1 className="font-heading text-2xl sm:text-3xl font-bold mb-3">
          Severity Analysis
        </h1>
        <p className="text-sm text-text-secondary leading-relaxed max-w-xl mb-10">
          After detection, each road segment is assigned a composite severity
          index. This page will visualise damage density along a selected route
          and export a prioritised repair schedule.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10">
          {[
            { tier: 'Minor', range: '0 – 0.3', color: '#2dd4bf', desc: 'Cosmetic damage. Monitor during next survey cycle.' },
            { tier: 'Moderate', range: '0.3 – 0.7', color: '#e8762a', desc: 'Schedule repair within 30 days to prevent escalation.' },
            { tier: 'Critical', range: '0.7 – 1.0', color: '#ef4444', desc: 'Immediate intervention required. Safety hazard.' },
          ].map((t) => (
            <div key={t.tier} className="border border-border rounded-lg p-5 bg-surface">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: t.color }} />
                <span className="font-heading text-sm font-semibold">{t.tier}</span>
              </div>
              <p className="text-xs text-text-muted leading-relaxed mb-3">{t.desc}</p>
              <span className="text-xs font-mono text-text-muted">Index: {t.range}</span>
            </div>
          ))}
        </div>

        <div className="border border-border rounded-lg bg-surface overflow-hidden">
          <div className="p-5 border-b border-border">
            <h2 className="text-xs font-medium text-text-muted uppercase tracking-widest">
              Damage Heatmap
            </h2>
          </div>
          <div className="h-72 sm:h-96 flex items-center justify-center bg-surface-2">
            <div className="text-center">
              <svg className="w-10 h-10 text-text-muted mx-auto mb-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 6.75V15m6-6v8.25m.503 3.498l4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 00-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0z" />
              </svg>
              <p className="text-sm text-text-muted">
                Interactive Folium map will render here.
              </p>
              <p className="text-xs text-text-muted mt-1">
                Requires processed detection results with GPS coordinates.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
