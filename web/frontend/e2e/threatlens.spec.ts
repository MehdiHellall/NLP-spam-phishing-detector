import { expect, test } from "@playwright/test";

test("classifies phish, ham, and spam with the real backend", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "ThreatLens" })).toBeVisible();
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("Test accuracy")).toBeVisible();

  await page.getByLabel("Message").fill("Urgent password reset required verify account.");
  await page.getByRole("button", { name: "Analyze" }).click();
  await expect(page.getByTestId("prediction-label")).toHaveText("Phishing", { timeout: 60_000 });
  await expect(page.getByTestId("prediction-risk")).toHaveText("High risk");
  await expect(page.getByText("Do not click links or share credentials")).toBeVisible();

  await page.getByRole("button", { name: "ham" }).click();
  await page.getByRole("button", { name: "Analyze" }).click();
  await expect(page.getByTestId("prediction-label")).toHaveText("Legitimate", {
    timeout: 60_000,
  });
  await expect(page.getByTestId("prediction-risk")).toHaveText("Low risk");

  await page.getByRole("button", { name: "spam" }).click();
  await page.getByRole("button", { name: "Analyze" }).click();
  await expect(page.getByTestId("prediction-label")).toHaveText("Spam", { timeout: 60_000 });
  await expect(page.getByTestId("prediction-risk")).toHaveText("Medium risk");
});

test("does not create horizontal overflow at the tested viewport", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 120_000 });

  const hasHorizontalOverflow = await page.evaluate(
    () => document.body.scrollWidth > document.documentElement.clientWidth + 1,
  );

  expect(hasHorizontalOverflow).toBe(false);
});
