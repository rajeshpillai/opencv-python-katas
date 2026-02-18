/**
 * demo-panel.tsx — Interactive demo controls (MVP: description + tips).
 * Future: sliders/toggles mapped to OpenCV parameters.
 */

import { Component, For, Show } from "solid-js";
import type { KataDetail } from "../api/client";

interface Props {
    kata: KataDetail;
}

const DemoPanel: Component<Props> = (props) => {
    return (
        <div class="demo-panel">
            <section class="demo-section">
                <h2 class="demo-section-title">What you'll learn</h2>
                <p class="demo-description">{props.kata.description}</p>
            </section>

            <Show when={props.kata.prerequisites.length > 0}>
                <section class="demo-section">
                    <h2 class="demo-section-title">Prerequisites</h2>
                    <ul class="demo-prereq-list">
                        <For each={props.kata.prerequisites}>
                            {(prereq) => (
                                <li class="demo-prereq-item">
                                    <span class="demo-prereq-icon">→</span>
                                    {prereq}
                                </li>
                            )}
                        </For>
                    </ul>
                </section>
            </Show>

            <Show when={props.kata.tips.length > 0}>
                <section class="demo-section">
                    <h2 class="demo-section-title">Tips & Common Mistakes</h2>
                    <ul class="demo-tips-list">
                        <For each={props.kata.tips}>
                            {(tip) => (
                                <li class="demo-tip-item">
                                    <span class="demo-tip-bullet">💡</span>
                                    <span>{tip}</span>
                                </li>
                            )}
                        </For>
                    </ul>
                </section>
            </Show>
        </div>
    );
};

export default DemoPanel;
