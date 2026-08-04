# Setup Notes

The main setup steps are kept in the project README.

The project has three parts:

- The Python model updates `data/marketData.json`.
- The Express server reads that data and handles login.
- The React client displays the dashboard.

You can start all three parts on Windows by running `run_app.bat` from the project root.

The batch file installs missing Node.js packages before starting the server and client. Python packages must be installed first with:

```powershell
pip install -r ml\requirements.txt
```
