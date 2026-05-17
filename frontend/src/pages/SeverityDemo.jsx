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
               <iframe
                  src="/severity_map.html"
                  title="Severity Map"
                  className="w-full h-full border-0"
                />
              
          </div>
        </div>
      </div>
    </div>
  );
}
