# Copilot Instructions for Playwright Web-Scraping Project

## Project Overview
This project uses [Playwright](https://playwright.dev/) for browser automation and end-to-end testing. The main focus is on web-scraping and automated browser testing, with all test code located in the `tests/` directory. The configuration is managed via `playwright.config.js`.

## Key Files and Structure
- `playwright.config.js`: Central Playwright configuration (testDir, parallelism, retries, reporters, device projects).
- `tests/`: Contains all Playwright test specs (e.g., `example.spec.js`).
- `.github/workflows/playwright.yml`: GitHub Actions workflow for CI (runs Playwright tests on push/PR).
- `playwright-report/`: HTML reports generated after test runs.
- `test-results/`: Raw test result artifacts.
- `package.json`: Declares dependencies (`@playwright/test`, `@types/node`).

## Developer Workflows
- **Install dependencies:** `npm ci` (CI) or `npm install` (local)
- **Run tests locally:** `npx playwright test`
- **View HTML report:** `npx playwright show-report` or open `playwright-report/index.html`
- **Install browsers:** `npx playwright install --with-deps`
- **CI:** Tests run automatically on push/PR to `main` or `master` via GitHub Actions.

## Project Conventions
- All tests must be placed in the `tests/` directory.
- Use Playwright's `test` and `expect` APIs for assertions and browser actions.
- Prefer using device/project profiles from `playwright.config.js` (e.g., `chromium`, `firefox`, `webkit`).
- Do not commit `.only` tests; CI will fail if `test.only` is present.
- Use the HTML reporter for reviewing test results.
- Environment variables can be loaded via dotenv (see commented code in `playwright.config.js`).

## Examples
- See `tests/example.spec.js` for Playwright test structure and usage.
- Example test:
  ```js
  test('has title', async ({ page }) => {
    await page.goto('https://playwright.dev/');
    await expect(page).toHaveTitle(/Playwright/);
  });
  ```

## External Integrations
- GitHub Actions for CI/CD (see `.github/workflows/playwright.yml`).
- Playwright HTML reports for test result visualization.

---

For more details, see the Playwright documentation: https://playwright.dev/
