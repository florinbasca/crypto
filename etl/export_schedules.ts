/**
 * Export per-protocol unlock/emission schedules to JSON for etl/unlocks.py.
 *
 * Usage: npx ts-node --transpile-only export_schedules.ts p1,p2,... out.json
 * (run from inside vendor/emissions-adapters so its imports resolve)
 *
 * Output: { [protocol]: { sections: [{label, category, series: [[unix_s,
 * cumulative_amount], ...]}], dropped?: string[], error?: string } }
 *
 * Per-section salvage: one broken section fetch (usually a staking-rewards
 * feed - AAVE's stkaave-incentives, dYdX's safety-module-rewards, Pendle's
 * ethereum-market-rewards) must not sink the protocol's investor/team
 * vesting cliffs, which are the sections the unlock features actually care
 * about. On failure, each section is probed independently and the export
 * reruns with the survivors; dropped sections are reported in the output.
 * Every attempt uses a FRESH adapter instance (require-cache bust):
 * createRawSections mutates the adapter's meta in place
 * (incompleteSections.lastRecord is resolved from a function to a value),
 * so reusing one instance across attempts fails spuriously.
 */
import "dotenv/config";
import fs from "fs";
import { createChartData } from "./utils/convertToChartData";
import { createRawSections } from "./utils/convertToRawData";

const META_KEYS = ["meta", "categories", "documented"];

function freshAdapter(protocol: string): any {
  const path = require.resolve(`./protocols/${protocol}`);
  delete require.cache[path];
  return require(path).default;
}

function subsetAdapter(adapter: any, sections: string[]): any {
  const sub: any = {};
  for (const k of sections) sub[k] = adapter[k];
  for (const m of META_KEYS) if (m in adapter) sub[m] = adapter[m];
  return sub;
}

async function buildSections(protocol: string, adapter: any): Promise<any[]> {
  const rawData = await createRawSections(adapter);
  const replaces = (adapter as any).documented?.replaces ?? [];
  const { realTimeData } = await createChartData(protocol, rawData, replaces);
  // realTimeData: [{section: string, data: {timestamps[], unlocked[],
  // isContinuous}}] (see utils/convertToChartData.ts)
  return realTimeData.map((s: any) => ({
    label: s.section,
    continuous: !!s.data?.isContinuous,
    series: (s.data?.timestamps ?? []).map((t: number, i: number) =>
      [t, s.data.unlocked[i]]),
  }));
}

async function exportOne(protocol: string): Promise<any> {
  try {
    return { sections: await buildSections(protocol, freshAdapter(protocol)) };
  } catch (primary: any) {
    const sectionKeys = Object.keys(freshAdapter(protocol)).filter(
      (k) => !META_KEYS.includes(k));
    if (sectionKeys.length < 2) throw primary; // nothing to salvage (or V2 shape)

    const good: string[] = [];
    const dropped: string[] = [];
    for (const k of sectionKeys) {
      try {
        await createRawSections(subsetAdapter(freshAdapter(protocol), [k]));
        good.push(k);
      } catch {
        dropped.push(k);
      }
    }
    if (!good.length || !dropped.length) throw primary;

    const sections = await buildSections(
      protocol, subsetAdapter(freshAdapter(protocol), good));
    console.log(`warn  ${protocol}: dropped section(s): ${dropped.join(", ")}`);
    return { sections, dropped };
  }
}

async function main() {
  const names = process.argv[2].split(",");
  const outPath = process.argv[3];
  const out: Record<string, any> = {};
  for (const name of names) {
    try {
      out[name] = await exportOne(name);
      console.log(`ok    ${name} (${out[name].sections.length} sections`
        + (out[name].dropped ? `, ${out[name].dropped.length} dropped)` : ")"));
    } catch (e: any) {
      out[name] = { error: String(e?.message ?? e).slice(0, 200) };
      console.log(`FAIL  ${name}: ${out[name].error.slice(0, 80)}`);
    }
  }
  fs.writeFileSync(outPath, JSON.stringify(out));
  const ok = Object.values(out).filter((v: any) => !v.error).length;
  console.log(`exported ${ok}/${names.length} protocols -> ${outPath}`);
}

main();
