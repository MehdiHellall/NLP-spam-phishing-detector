import { expect, test } from "@playwright/test";

const PHISHING_MESSAGE = "Urgent password reset required verify account.";
const PERCENTAGE = /^\d+(?:\.\d+)?%$/;

test.beforeEach(async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "ThreatLens" })).toBeVisible();
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 120_000 });
});

test("keeps the real classifier as the only primary action", async ({ page }) => {
  await expect(page.getByRole("button")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Analyze", exact: true })).toBeVisible();

  for (const exampleName of ["ham", "phish", "spam"]) {
    await expect(page.getByRole("button", { name: exampleName, exact: true })).toHaveCount(0);
  }

  await expect(page.getByLabel("Message")).toBeVisible();
  await expect(page.getByText(/not stored/i).first()).toBeVisible();
});

test("submits direct analyst input to the real v1 model and renders the complete result", async ({
  page,
}) => {
  const textarea = page.getByLabel("Message");
  await textarea.fill(PHISHING_MESSAGE);
  await expect(textarea).toHaveValue(PHISHING_MESSAGE);

  const predictionResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/v1/predict",
  );

  await page.getByRole("button", { name: "Analyze", exact: true }).click();
  const predictionResponse = await predictionResponsePromise;
  expect(predictionResponse.ok()).toBe(true);

  await expect(page.getByTestId("prediction-label")).toHaveText("Phishing", {
    timeout: 60_000,
  });
  await expect(page.getByTestId("prediction-risk")).toHaveText("High risk");
  await expect(page.getByTestId("prediction-confidence")).toHaveText(PERCENTAGE);

  for (const label of ["ham", "phish", "spam"]) {
    await expect(page.getByTestId(`probability-${label}`)).toHaveText(PERCENTAGE);
  }

  await expect(page.getByTestId("prediction-explanation")).toContainText(
    "trained model classified",
  );
  await expect(page.getByTestId("suggested-action")).toContainText(
    "Do not click links or share credentials",
  );
  await expect(page.getByTestId("model-artifact")).toContainText("tfidf_logreg.joblib");
});

test("shows a useful error when the model is temporarily unavailable", async ({ page }) => {
  const errorDetail = "Model artifact is temporarily unavailable. Try again shortly.";

  await page.route("**/v1/predict", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }

    await route.fulfill({
      status: 503,
      contentType: "application/json",
      headers: { "access-control-allow-origin": "*" },
      body: JSON.stringify({ detail: errorDetail }),
    });
  });

  await page.getByLabel("Message").fill(PHISHING_MESSAGE);
  await page.getByRole("button", { name: "Analyze", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText(errorDetail);
  await expect(page.getByRole("button", { name: "Analyze", exact: true })).toBeEnabled();
});

test("keeps analysis available after a transient readiness failure", async ({ page }) => {
  await page.route("**/v1/ready", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      headers: { "access-control-allow-origin": "*" },
      body: JSON.stringify({
        status: "error",
        model_loaded: false,
        model_path: "tfidf_logreg.joblib",
        detail: "Model artifact is temporarily unavailable.",
      }),
    });
  });

  await page.reload();
  await expect(page.getByText("Model offline")).toBeVisible();
  await page.getByLabel("Message").fill(PHISHING_MESSAGE);

  await expect(page.getByRole("button", { name: "Analyze", exact: true })).toBeEnabled();
});

test("does not create horizontal overflow at the tested viewport", async ({ page }) => {
  const hasHorizontalOverflow = await page.evaluate(
    () => document.body.scrollWidth > document.documentElement.clientWidth + 1,
  );

  expect(hasHorizontalOverflow).toBe(false);
});
