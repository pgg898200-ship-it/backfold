// From the repository root: make && node tests/demo.cjs
"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const {pack, unpack, samples} = require("../demo/app.js");
const program = path.resolve(__dirname, "../build/eventfold");
const exampleStreams = Object.values(samples()).map(value => Buffer.from(value));
const cases = [
  Buffer.alloc(0), Buffer.from("\n\n\r\r\n\r\n\n"), Buffer.from([0, 255, 13, 13, 254]),
  ...exampleStreams,
];
function addFixtures(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) addFixtures(file);
    else if (entry.name.endsWith(".sse")) cases.push(fs.readFileSync(file));
  }
}
addFixtures(path.resolve(__dirname, "../fixtures"));
let seed = 199743;
function random(n) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed % n;
}
const randomFrames = [];
for (let i = 0; i < 1000; i++) {
  const text = Buffer.alloc(random(580) + 1);
  for (let j = 0; j < text.length; j++) text[j] = 32 + random(93);
  const before = Buffer.concat([Buffer.from("data: "), text, Buffer.from("\n\n")]);
  const after = Buffer.from(before);
  after[6 + random(text.length)] = 126;
  after[6 + random(text.length)] = 127;
  randomFrames.push(before, after);
  if (randomFrames.length === 200) {
    cases.push(Buffer.concat(randomFrames));
    randomFrames.length = 0;
  }
}
for (const raw of cases) {
  const wire = Buffer.from(pack(raw).wire);
  const native = spawnSync(program, ["pack"], {input: raw});
  assert.equal(native.status, 0, native.error?.message || native.stderr?.toString());
  assert.deepEqual(wire, native.stdout);
  assert.deepEqual(Buffer.concat(unpack(native.stdout).map(r => r.bytes)), raw);
}
const kinds = new Set(exampleStreams.flatMap(data => unpack(pack(data).wire).map(r => r.kind)));
assert.deepEqual([...kinds].sort(), ["Literal", "Patch", "Repeat", "Two-splice"]);
for (const badRecord of [[255], [127, 128, 0, 0, 0], [129, 2, 0, 0, 1, 0, 0], [64, 128], [4, 1]]) {
  assert.throws(() => unpack(Buffer.from([...Buffer.from("EFL1"), ...badRecord])));
}
console.log("Browser and native EventFold wire bytes match, on all bundled SSE fixtures and 2,000 randomized adjacent frames.");
