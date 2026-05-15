import { Link } from 'react-router-dom';

export default function DetectDemo() {
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
          Pothole Detection
        </h1>
        <p className="text-sm text-text-secondary leading-relaxed max-w-xl mb-10">
          Upload a road image or paste a video frame below. The YOLOv8 model
          will run inference and return bounding boxes around detected defects
          with confidence scores and class labels.
        </p>

        <div className="border-2 border-dashed border-border rounded-lg p-10 sm:p-16 flex flex-col items-center justify-center text-center mb-8">
          <svg className="w-10 h-10 text-text-muted mb-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16v-8m0 0l-3 3m3-3l3 3M4.5 19.5h15a1.5 1.5 0 001.5-1.5v-12a1.5 1.5 0 00-1.5-1.5h-15A1.5 1.5 0 003 6v12a1.5 1.5 0 001.5 1.5z" />
          </svg>
          <p className="text-sm text-text-muted mb-1">
            Drag and drop an image, or click to browse
          </p>
          <p className="text-xs text-text-muted">
            JPG, PNG up to 10 MB — inference takes ~200 ms on GPU
          </p>
        </div>

        <div className="border border-border rounded-lg p-6 bg-surface">
          <h2 className="text-xs font-medium text-text-muted uppercase tracking-widest mb-4">
            Detection Results
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            {['D00 — Long. Crack', 'D10 — Trans. Crack', 'D20 — Alligator', 'D40 — Pothole'].map((cls) => (
              <div key={cls} className="bg-surface-2 border border-border rounded-md py-4 px-3">
                <div className="font-heading text-xl font-semibold text-text-muted">—</div>
                <div className="text-xs text-text-muted mt-1">{cls}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-text-muted mt-5 text-center">
            Results will appear here after an image is uploaded. Model backend coming soon.
          </p>
        </div>
      </div>
    </div>
  );
}
