import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DeleteMemoryButton } from "@/components/memories/delete-memory-button";
import { api } from "@/lib/api";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

afterEach(() => vi.restoreAllMocks());

function renderWithClient(ui: React.ReactNode) {
  const qc = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe("DeleteMemoryButton", () => {
  it("requires explicit confirmation before deleting", async () => {
    const spy = vi
      .spyOn(api, "deleteMemory")
      .mockResolvedValue({ deleted_ids: [7], count: 1 });
    const user = userEvent.setup();
    const onDeleted = vi.fn();

    renderWithClient(<DeleteMemoryButton id={7} onDeleted={onDeleted} />);

    // Nothing happens until the confirm dialog is opened and confirmed.
    expect(spy).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /delete episode 7/i }));
    expect(
      await screen.findByText(/delete episode #7\?/i),
    ).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(spy).toHaveBeenCalledWith(7));
    await waitFor(() => expect(onDeleted).toHaveBeenCalled());
  });
});
