/**
 * kata-sidebar.tsx — Left sidebar listing all katas grouped by level.
 */

import { Component, For, createMemo } from "solid-js";
import { A } from "@solidjs/router";
import type { KataListItem } from "../api/client";

interface Props {
    katas: KataListItem[];
    currentSlug: string;
}

const LEVEL_ORDER = ["beginner", "intermediate", "advanced"] as const;

const LEVEL_LABELS: Record<string, string> = {
    beginner: "Beginner",
    intermediate: "Intermediate",
    advanced: "Advanced",
};

const KataSidebar: Component<Props> = (props) => {
    const grouped = createMemo(() => {
        const map: Record<string, KataListItem[]> = {};
        for (const kata of props.katas) {
            if (!map[kata.level]) map[kata.level] = [];
            map[kata.level].push(kata);
        }
        return map;
    });

    return (
        <aside class="sidebar">
            <div class="sidebar-header">
                <span class="sidebar-logo">⚡</span>
                <span class="sidebar-title">OpenCV Playground</span>
            </div>

            <nav class="sidebar-nav">
                <For each={LEVEL_ORDER}>
                    {(level) =>
                        grouped()[level]?.length ? (
                            <div class="sidebar-group">
                                <div class={`sidebar-group-label sidebar-group-label--${level}`}>
                                    {LEVEL_LABELS[level]}
                                </div>
                                <For each={grouped()[level]}>
                                    {(kata) => (
                                        <A
                                            href={`/kata/${kata.slug}`}
                                            class="sidebar-item"
                                            classList={{ "sidebar-item--active": kata.slug === props.currentSlug }}
                                        >
                                            <span class="sidebar-item-title">{kata.title}</span>
                                            <div class="sidebar-item-concepts">
                                                <For each={kata.concepts.slice(0, 2)}>
                                                    {(c) => <span class="concept-tag">{c}</span>}
                                                </For>
                                            </div>
                                        </A>
                                    )}
                                </For>
                            </div>
                        ) : null
                    }
                </For>
            </nav>
        </aside>
    );
};

export default KataSidebar;
