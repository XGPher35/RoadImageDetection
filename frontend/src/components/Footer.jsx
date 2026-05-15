export default function Footer() {
  return (
    <footer className="py-8 border-t border-border/50">
      <div className="site-container flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 text-sm text-text-muted">
        <span>SHP — Smart Highway Patrol</span>
        <span>© {new Date().getFullYear()}</span>
      </div>
    </footer>
  );
}
