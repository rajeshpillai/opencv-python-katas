/**
 * kata-page.tsx — Main kata view with two tabs: Details and Code.
 */

import { Component, createSignal, createResource, Show } from "solid-js";
import { useParams } from "@solidjs/router";
import { api } from "../api/client";
import type { ExecuteResult } from "../api/client";
import KataHeader from "../components/kata-header";
import DemoPanel from "../components/demo-panel";
import CodeEditor from "../components/code-editor";
import OutputPanel from "../components/output-panel";

type Tab = "details" | "code";

const KataPage: Component = () => {
    const params = useParams<{ slug: string }>();
    const [kata] = createResource(() => params.slug, api.getKata);

    const [activeTab, setActiveTab] = createSignal<Tab>("details");
    const [code, setCode] = createSignal("");
    const [result, setResult] = createSignal<ExecuteResult | null>(null);
    const [running, setRunning] = createSignal(false);

    const starterCode = () => kata()?.starter_code ?? "";

    const handleReset = () => {
        setCode(starterCode());
        setResult(null);
    };

    const handleRun = async () => {
        if (running()) return;
        setRunning(true);
        setResult(null);
        try {
            const res = await api.executeCode(code() || starterCode());
            setResult(res);
        } catch (e: any) {
            setResult({ image_b64: null, logs: "", error: e.message });
        } finally {
            setRunning(false);
        }
    };

    return (
        <div class="kata-page">
            <Show when={kata.loading}>
                <div class="kata-loading">Loading kata…</div>
            </Show>

            <Show when={kata.error}>
                <div class="kata-error">Failed to load kata: {String(kata.error)}</div>
            </Show>

            <Show when={kata()}>
                {(k) => (
                    <>
                        <KataHeader kata={k()} />

                        {/* Tab bar */}
                        <div class="kata-tabs">
                            <button
                                class="kata-tab"
                                classList={{ "kata-tab--active": activeTab() === "details" }}
                                onClick={() => setActiveTab("details")}
                            >
                                📖 Details
                            </button>
                            <button
                                class="kata-tab"
                                classList={{ "kata-tab--active": activeTab() === "code" }}
                                onClick={() => setActiveTab("code")}
                            >
                                💻 Code
                            </button>
                        </div>

                        {/* Tab content */}
                        <div class="kata-tab-content">
                            <Show when={activeTab() === "details"}>
                                <DemoPanel kata={k()} />
                            </Show>

                            <Show when={activeTab() === "code"}>
                                <div class="kata-code-layout">
                                    {/* Editor pane */}
                                    <div class="kata-editor-pane">
                                        <div class="kata-editor-toolbar">
                                            <span class="kata-editor-filename">kata.py</span>
                                            <div class="kata-editor-actions">
                                                <button class="btn btn--ghost" onClick={handleReset}>
                                                    ↺ Reset
                                                </button>
                                                <button
                                                    class="btn btn--primary"
                                                    onClick={handleRun}
                                                    disabled={running()}
                                                >
                                                    {running() ? "Running…" : "▶ Run"}
                                                </button>
                                            </div>
                                        </div>
                                        <CodeEditor
                                            value={code() || starterCode()}
                                            onChange={setCode}
                                        />
                                    </div>

                                    {/* Output pane */}
                                    <OutputPanel result={result()} loading={running()} />
                                </div>
                            </Show>
                        </div>
                    </>
                )}
            </Show>
        </div>
    );
};

export default KataPage;
