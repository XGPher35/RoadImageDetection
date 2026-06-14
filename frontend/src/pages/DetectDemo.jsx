import { useState, useCallback } from 'react';
import Navbar from '../components/Navbar';
import ImageUpload from '../components/ImageUpload';

const CLASS_INFO = [
  { key: 'Longitudinal Crack', label: 'Longitudinal', color: '#3498db' },
  { key: 'Transverse Crack',   label: 'Transverse',   color: '#2dd4bf' },
  { key: 'Alligator Crack',    label: 'Alligator',    color: '#e67e22' },
  { key: 'Pothole',            label: 'Pothole',      color: '#e74c3c' },
];

export default function DetectDemo() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/detect', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-base text-text-primary">
      <Navbar />

      <div className="site-container pt-24 pb-16">
        {/* Page header */}
        <div className="mb-10">
          <p className="text-xs font-semibold text-accent uppercase tracking-widest mb-3">Detect</p>
          <h1 className="font-heading text-3xl sm:text-4xl font-bold mb-4">
            Road Damage Detection
          </h1>
          <p className="text-sm text-text-secondary leading-relaxed max-w-xl">
            Upload a road image to identify surface defects. The YOLOv8 model classifies
            each detection into one of four damage categories and returns bounding boxes
            with confidence scores.
          </p>
        </div>

        <ImageUpload onFileSelected={handleUpload} disabled={loading} />

        {loading && (
          <div className="flex items-center justify-center gap-3 py-16">
            <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-text-muted">Analysing image…</span>
          </div>
        )}

        {error && (
          <div className="mt-6 border border-red-500/30 bg-red-500/5 rounded-xl px-5 py-4 text-sm text-red-400">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-10 space-y-6">
            {/* Class count cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {CLASS_INFO.map((cls) => {
                const count = result.class_counts?.[cls.key] || 0;
                return (
                  <div
                    key={cls.key}
                    className="border rounded-xl py-5 px-4 text-center transition-all"
                    style={{
                      borderColor: count > 0 ? `${cls.color}40` : 'var(--color-border)',
                      background: count > 0 ? `${cls.color}0a` : 'var(--color-surface)',
                    }}
                  >
                    <div
                      className="font-heading text-3xl font-bold tabular-nums"
                      style={{ color: count > 0 ? cls.color : 'var(--color-text-muted)' }}
                    >
                      {count}
                    </div>
                    <div className="text-xs text-text-muted mt-1.5">{cls.label}</div>
                  </div>
                );
              })}
            </div>

            {/* Annotated image */}
            {result.image && (
              <div className="border border-border rounded-xl overflow-hidden">
                <div className="px-5 py-3.5 border-b border-border flex items-center justify-between bg-surface">
                  <span className="text-xs font-semibold text-text-muted uppercase tracking-widest">
                    Annotated Output
                  </span>
                  <span className="text-xs text-text-muted font-mono">
                    {result.image_width}×{result.image_height} · {result.total_detections} defect{result.total_detections !== 1 ? 's' : ''}
                  </span>
                </div>
                <div className="p-4 flex justify-center bg-surface-2">
                  <img
                    src={`data:image/jpeg;base64,${result.image}`}
                    alt="Detection result with annotated bounding boxes"
                    className="max-w-full max-h-[520px] rounded-lg object-contain"
                  />
                </div>
              </div>
            )}

            {/* Detections table */}
            {result.detections?.length > 0 && (
              <div className="border border-border rounded-xl overflow-hidden">
                <div className="px-5 py-3.5 border-b border-border bg-surface">
                  <span className="text-xs font-semibold text-text-muted uppercase tracking-widest">
                    Detections
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-text-muted uppercase tracking-wider border-b border-border bg-surface/60">
                        <th className="px-5 py-3 w-12">#</th>
                        <th className="px-5 py-3">Class</th>
                        <th className="px-5 py-3 text-right">Confidence</th>
                        <th className="px-5 py-3 text-right hidden sm:table-cell">Area (px²)</th>
                        <th className="px-5 py-3 text-right">Frame %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((d, i) => (
                        <tr key={i} className="border-b border-border/40 hover:bg-surface/40 transition-colors">
                          <td className="px-5 py-3 font-mono text-text-muted text-xs">{i + 1}</td>
                          <td className="px-5 py-3">
                            <span className="inline-flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: CLASS_INFO.find(c => c.key === d.class_name)?.color || '#888' }} />
                              {d.class_name}
                            </span>
                          </td>
                          <td className="px-5 py-3 text-right font-mono">{(d.confidence * 100).toFixed(1)}%</td>
                          <td className="px-5 py-3 text-right font-mono hidden sm:table-cell">{Math.round(d.bbox_area).toLocaleString()}</td>
                          <td className="px-5 py-3 text-right font-mono">{(d.relative_area * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {result.total_detections === 0 && (
              <div className="border border-teal/20 bg-teal/5 rounded-xl px-6 py-6 text-center">
                <p className="text-sm text-teal font-semibold mb-1">No defects detected</p>
                <p className="text-xs text-text-muted">
                  This road surface appears to be in good condition. Try uploading an image with visible damage.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
