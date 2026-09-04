export default function Header() {
  return (
    <header className="app-header">
      <div className="brand-block">
        <span className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="brand-name">AI Finance</span>
      </div>

      <div className="header-actions">
        <span className="research-label">Learning project</span>
        <span className="advice-label">Not financial advice</span>
      </div>
    </header>
  );
}
