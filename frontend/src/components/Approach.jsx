import { useInView } from '../hooks/useInView';

const cards = [
  {
    title: 'Experiments Done So Far',
    body: 'We have completed the initial dataset audit, establishing the total images, class distributions, and train/validation splits. We have generated augmented samples to showcase in the report. A baseline YOLO model has been trained for 20-30 epochs to extract the training loss curve and baseline mAP@50 metrics per class.',
    detail: 'Additionally, we have formulated the severity index, assigning class weights, and evaluated it against sample YOLO predictions. A Folium map prototype has been developed with initial markers representing collected local data.',
  },
  {
    title: 'Upcoming: Model Optimisation',
    body: 'Our upcoming experiments focus on advanced augmentation techniques not yet applied, such as simulated rain, nighttime, and motion blur. We plan to address class imbalances identified during the audit through oversampling or weighted loss functions.',
    detail: 'We will also explore hyperparameter tuning (e.g., learning rate schedules, anchor sizes) and compare model variants (YOLOv8m vs YOLOv8l) to surpass our target baseline mAP@50.',
  },
  {
    title: 'Upcoming: Validation & Generalisation',
    body: 'We will design a validation experiment for the severity index to ensure the rankings match human judgment, establishing clear test cases and ground truths.',
    detail: 'A dedicated experiment for the local dataset will outline target image counts, annotation plans, and a strategy to measure model generalisation (calculating the mAP drop when moving from RDD2022 to local road conditions).',
  },
];

export default function Approach() {
  const [ref, inView] = useInView();

  return (
    <section
      id="approach"
      ref={ref}
      className={`py-24 lg:py-32 bg-surface border-t border-border/50 transition-all duration-700 ${
        inView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}
    >
      <div className="site-container">
        <p className="text-sm font-medium text-accent uppercase tracking-widest mb-6">
          Methodology & Experiments
        </p>
        <h2 className="font-heading text-3xl sm:text-4xl font-semibold leading-tight mb-12">
          Midterm progress and experimental design.
        </h2>

        <div className="space-y-6">
          {cards.map((c, i) => (
            <div
              key={i}
              className="bg-surface-2 border border-border rounded-xl overflow-hidden p-8"
            >
              <div className="flex items-center gap-4 mb-6">
                <span className="shrink-0 w-10 h-10 rounded-lg bg-base border border-border flex items-center justify-center text-sm text-text-muted font-mono">
                  {i + 1}
                </span>
                <h3 className="font-heading text-xl font-semibold">
                  {c.title}
                </h3>
              </div>
              <div className="text-base text-text-secondary leading-relaxed space-y-4">
                <p>{c.body}</p>
                <p className="text-text-muted">{c.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
