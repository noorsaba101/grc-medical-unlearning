const test = require("node:test");
const assert = require("node:assert/strict");
const {ethers} = require("ethers");
const {compile, run} = require("../scripts/evm_bridge.cjs");

test("Solidity compiles offline with pinned solc-js", () => {
  assert.ok(compile().bytecode.length > 100);
});

test("real EVM enforces actors, ordering, pass/fail, immutability and compact hashes", {
  skip: !process.env.GRC_EVM_RPC, timeout: 45000
}, async () => {
  const rpc_url = process.env.GRC_EVM_RPC;
  const deployment = await run({operation: "deploy", rpc_url});
  const provider = new ethers.JsonRpcProvider(rpc_url, undefined, {cacheTimeout: -1});
  provider.pollingInterval = 50;
  try {
    const [governor, operator, verifier, stranger] = await Promise.all([0,1,2,3].map(i => provider.getSigner(i)));
    const contract = new ethers.Contract(deployment.contract_address, compile().abi, provider);
    const d = n => ethers.sha256(ethers.toUtf8Bytes(n));
    const id = d("pass"), failedId = d("fail");
    await assert.rejects(contract.connect(governor).authorizeRequest(id));
    await assert.rejects(contract.connect(stranger).registerUnlearningRequest(id, d("request"), d("policy id"), d("policy"), d("source")));
    await (await contract.connect(operator).registerUnlearningRequest(id, d("request"), d("policy id"), d("policy"), d("source"))).wait();
    await assert.rejects(contract.connect(operator).registerUnlearningRequest(id, d("request"), d("policy id"), d("policy"), d("source")));
    await assert.rejects(contract.connect(operator).authorizeRequest(id));
    await assert.rejects(contract.connect(verifier).recordVerification(id, d("result"), d("verify"), true));
    await (await contract.connect(governor).authorizeRequest(id)).wait();
    await assert.rejects(contract.connect(governor).completeRequest(id));
    await assert.rejects(contract.connect(operator).recordVerification(id, d("result"), d("verify"), true));
    await (await contract.connect(verifier).recordVerification(id, d("result"), d("verify"), true)).wait();
    await (await contract.connect(governor).completeRequest(id)).wait();
    let row = await contract.getRequest(id);
    assert.equal(Number(row.status), 5);
    assert.equal(row.sourceModelHash, d("source"));
    assert.equal(row.resultModelHash, d("result"));
    assert.equal(row.verificationHash, d("verify"));
    assert.ok(row.registeredAt <= row.authorizedAt && row.authorizedAt <= row.verifiedAt && row.verifiedAt <= row.completedAt);
    await assert.rejects(contract.connect(verifier).recordVerification(id, d("other"), d("other"), false));
    await (await contract.connect(operator).registerUnlearningRequest(failedId, d("r2"), d("p2"), d("p2"), d("s2"))).wait();
    await (await contract.connect(governor).authorizeRequest(failedId)).wait();
    await (await contract.connect(verifier).recordVerification(failedId, d("result2"), d("v2"), false)).wait();
    await assert.rejects(contract.connect(governor).completeRequest(failedId));
    row = await contract.getRequest(failedId);
    assert.equal(Number(row.status), 4);
  } finally { provider.destroy(); }
});
