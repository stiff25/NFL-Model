NFL MODEL — AUTOMATIC VERSION

This removes the weekly manual CSV step.

1. Create a GitHub repository (for example nfl-model).
2. Upload the contents of this package.
3. GitHub -> Settings -> Secrets and variables -> Actions.
4. Add repository secret ODDS_API_KEY with your The Odds API key.
5. GitHub -> Actions -> Update NFL Model -> Run workflow once.
6. Enable GitHub Pages from the main branch/root.
7. Open the Pages URL on Android Chrome and add it to the home screen.

The workflow runs every 3 hours and publishes data/current_picks.json.
The dashboard reads that feed automatically.

The automation does not prove or guarantee a 62% historical win rate; that
must be established by a strict walk-forward backtest.
