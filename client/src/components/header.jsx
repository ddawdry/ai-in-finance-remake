export default function Header({ onLogout }) {
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
        <button type="button" className="logout-button" onClick={onLogout}>
          Log out
        </button>
      </div>
    </header>
  );
}
