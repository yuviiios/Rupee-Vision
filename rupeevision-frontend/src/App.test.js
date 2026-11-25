import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }) => children,
  LineChart: ({ data }) => JSON.stringify(data),
  Line: () => null,
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null
}));

afterEach(() => jest.restoreAllMocks());

test.each([
  ["Predict", "/predict"],
  ["Predict with latest data", "/predict?latest=1"]
])("%s shows values against the dates returned by the API", async (button, endpoint) => {
  const currencies = ["US Dollar", "Euro", "Japanese Yen", "Pound Sterling"];
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      dates: ["2025-03-29"],
      as_of: "2025-03-28",
      predictions: Object.fromEntries(currencies.map(name => [name, { values: [85.5] }]))
    })
  });
  render(<App />);
  fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "1" } });
  fireEvent.click(screen.getByRole("button", { name: button }));
  expect(global.fetch.mock.calls[0][0].endsWith(endpoint)).toBe(true);
  expect(await screen.findByText(/2025-03-29/)).toHaveTextContent('"US Dollar":85.5');
  expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("2025-03-28");
});

test("reports an unavailable forecast", async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: false });
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Predict" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Forecast unavailable");
});
