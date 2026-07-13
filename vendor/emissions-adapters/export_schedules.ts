/**
 * Export per-protocol unlock/emission schedules to JSON for etl/unlocks.py.
 *
 * Usage: npx ts-node --transpile-only export_schedules.ts p1,p2,... out.json
 * (run from inside vendor/emissions-adapters so its imports resolve)
 *
 * Output: { [protocol]: { sections: [{label, category, series: [[unix_s,
 * cumulative_amount], ...]}], error?: string } }
 */
import "dotenv/config";
import fs from "fs";
import { createChartData } from "./utils/convertToChartData";
import { createRawSections } from "./utils/convertToRawData";

async function exportOne(protocol: string): Promise<any> {
  const adapter = require(`./protocols/${protocol}`).default;
  const rawData = await createRawSections(adapter);
  const replaces = (adapter as any).documented?.replaces ?? [];
  const { realTimeData } = await createChartData(protocol, rawData, replaces);
  // realTimeData: [{section: string, data: {timestamps[], unlocked[],
  // isContinuous}}] (see utils/convertToChartData.ts)
  const sections = realTimeData.map((s: any) => ({
    label: s.section,
    continuous: !!s.data?.isContinuous,
    series: (s.data?.timestamps ?? []).map((t: number, i: number) =>
      [t, s.data.unlocked[i]]),
  }));
  return { sections };
}

async function main() {
  const names = process.argv[2].split(",");
  const outPath = process.argv[3];
  const out: Record<string, any> = {};
  for (const name of names) {
    try {
      out[name] = await exportOne(name);
      console.log(`ok    ${name} (${out[name].sections.length} sections)`);
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
