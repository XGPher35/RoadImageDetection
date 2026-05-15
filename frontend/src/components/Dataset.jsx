import { useInView } from '../hooks/useInView';

export default function Dataset() {
  const [ref, inView] = useInView();

  return (
    <section
      id="dataset"
      ref={ref}
      className={`py-24 lg:py-32 border-t border-border/50 transition-all duration-700 ${
        inView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}
    >
      <div className="site-container">
        <p className="text-sm font-medium text-accent uppercase tracking-widest mb-6">
          Dataset
        </p>
        <h2 className="font-heading text-3xl sm:text-4xl font-semibold leading-tight mb-12">
          Trained on global benchmarks, validated on local roads.
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* RDD2022 */}
          <div className="border border-border rounded-xl p-8 flex flex-col">
            <h3 className="font-heading text-xl font-semibold mb-4">
              RDD2022
            </h3>
            <div className="space-y-4 text-base text-text-secondary leading-relaxed mb-8 flex-1">
              <p>
                The Road Damage Dataset 2022 was released as part of the
                Crowdsensing-based Road Damage Detection Challenge (CRDDC&rsquo;22).
                It contains 47,420 road images collected across six countries —
                Japan, India, Czech Republic, Norway, the United States, and China
                — with over 55,000 bounding-box annotations in PASCAL VOC format
                covering four damage categories (D00, D10, D20, D40).
              </p>
              <p>
                We use the official train/test split and augment the training
                partition with Albumentations (random crop, HSV jitter, horizontal
                flip) to improve generalisation to local road surfaces.
              </p>
            </div>
            <div className="flex flex-wrap gap-8 text-base border-t border-border pt-6 mt-auto">
              <div>
                <span className="font-heading font-semibold">47,420</span>{' '}
                <span className="text-text-muted">images</span>
              </div>
              <div>
                <span className="font-heading font-semibold">55K+</span>{' '}
                <span className="text-text-muted">annotations</span>
              </div>
              <div>
                <span className="font-heading font-semibold">6</span>{' '}
                <span className="text-text-muted">countries</span>
              </div>
            </div>
          </div>

          {/* Local Road Dataset */}
          <div className="border border-border rounded-xl p-8 flex flex-col">
            <h3 className="font-heading text-xl font-semibold mb-4">
              Local Road Dataset
            </h3>
            <div className="space-y-4 text-base text-text-secondary leading-relaxed mb-8 flex-1">
              <p>
                To supplement RDD2022 with specific conditions, we collected
                images along local highway corridors. A camera mounted
                on the dashboard of a survey vehicle recorded footage at 1080p,
                and key frames were extracted every 2 seconds.
              </p>
              <p>
                Annotators labelled each frame using LabelImg following the
                same four-class schema as RDD2022. Inter-annotator agreement
                (IoU ≥ 0.5) was calculated before adjudication. The final set
                includes images from both dry-season and post-monsoon conditions.
              </p>
            </div>
            <div className="flex flex-wrap gap-8 text-base border-t border-border pt-6 mt-auto">
              <div>
                <span className="font-heading font-semibold">2,500+</span>{' '}
                <span className="text-text-muted">images</span>
              </div>
              <div>
                <span className="font-heading font-semibold">4</span>{' '}
                <span className="text-text-muted">classes</span>
              </div>
              <div>
                <span className="font-heading font-semibold">2</span>{' '}
                <span className="text-text-muted">regions</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
