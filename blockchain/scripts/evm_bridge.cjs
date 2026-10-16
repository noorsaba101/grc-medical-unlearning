// JSON-only stdin/stdout bridge. The only chain arguments are compact hashes/booleans.
const fs = require("node:fs");
const path = require("node:path");
const solc = require("solc");
const { ethers } = require("ethers");
const statuses = ["NONE", "REGISTERED", "AUTHORIZED", "VERIFIED_PASS", "VERIFIED_FAIL", "COMPLETED"];

function compile() {
  const source = fs.readFileSync(path.join(__dirname, "../contracts/UnlearningGovernance.sol"), "utf8");
  const input = {language: "Solidity", sources: {"UnlearningGovernance.sol": {content: source}},
    settings: {optimizer: {enabled: true, runs: 200}, evmVersion: "shanghai",
      outputSelection: {"*": {"*": ["abi", "evm.bytecode.object"]}}}};
  const output = JSON.parse(solc.compile(JSON.stringify(input)));
  const errors = (output.errors || []).filter(e => e.severity === "error");
  if (errors.length) throw new Error(errors.map(e => e.formattedMessage).join("\n"));
  const artifact = output.contracts["UnlearningGovernance.sol"].UnlearningGovernance;
  return {abi: artifact.abi, bytecode: "0x" + artifact.evm.bytecode.object, compiler: solc.version()};
}

function localUrl(value) {
  const url = new URL(value);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
    throw new Error("Only loopback HTTP development RPC endpoints are permitted");
  }
  return value;
}

async function run(input) {
  const provider = new ethers.JsonRpcProvider(localUrl(input.rpc_url), undefined, {cacheTimeout: -1});
  provider.pollingInterval = 50;
  try {
    const network = await provider.getNetwork();
    if (Number(network.chainId) !== 31337) throw new Error("Only local development chain 31337 is permitted");
    const artifact = compile();
    if (input.operation === "deploy") {
      const governor = await provider.getSigner(0), operator = await provider.getSigner(1), verifier = await provider.getSigner(2);
      const contract = await new ethers.ContractFactory(artifact.abi, artifact.bytecode, governor)
        .deploy(await operator.getAddress(), await verifier.getAddress());
      await contract.waitForDeployment();
      const receipt = await contract.deploymentTransaction().wait();
      const block = await provider.getBlock(receipt.blockNumber);
      return {mode: "real_local_evm", chain_id: Number(network.chainId), contract_address: await contract.getAddress(),
        governor: await governor.getAddress(), operator: await operator.getAddress(), verifier: await verifier.getAddress(),
        transaction_hash: receipt.hash, block_number: receipt.blockNumber, timestamp: block.timestamp,
        compiler: artifact.compiler, runtime_code_hash: ethers.keccak256(await provider.getCode(await contract.getAddress()))};
    }
    if (!ethers.isAddress(input.contract_address) || await provider.getCode(input.contract_address) === "0x") {
      throw new Error("No governance contract at the supplied address");
    }
    const contract = new ethers.Contract(input.contract_address, artifact.abi, provider);
    if (input.operation === "get") {
      const r = await contract.getRequest(input.request_id);
      return {status: statuses[Number(r.status)], request_hash: r.requestHash, policy_id_hash: r.policyIdHash,
        policy_hash: r.policyHash, source_model_hash: r.sourceModelHash, result_model_hash: r.resultModelHash,
        verification_hash: r.verificationHash, registered_at: Number(r.registeredAt), authorized_at: Number(r.authorizedAt),
        verified_at: Number(r.verifiedAt), completed_at: Number(r.completedAt), requester: r.requester,
        authorizer: r.authorizer, verified_by: r.verifiedBy, completed_by: r.completedBy};
    }
    const methods = {register: ["registerUnlearningRequest", 1, [input.request_id, input.request_hash,
      input.policy_id_hash, input.policy_hash, input.source_model_hash]],
      authorize: ["authorizeRequest", 0, [input.request_id]],
      verify: ["recordVerification", 2, [input.request_id, input.result_model_hash, input.verification_hash, input.passed]],
      complete: ["completeRequest", 0, [input.request_id]]};
    if (!methods[input.operation]) throw new Error("Unknown operation");
    const [method, actor, args] = methods[input.operation];
    const signer = await provider.getSigner(actor);
    const tx = await contract.connect(signer)[method](...args);
    const receipt = await tx.wait();
    if (receipt.status !== 1) throw new Error("Transaction reverted");
    const block = await provider.getBlock(receipt.blockNumber);
    return {mode: "real_local_evm", operation: input.operation, actor: await signer.getAddress(),
      transaction_hash: receipt.hash, block_number: receipt.blockNumber, timestamp: block.timestamp};
  } finally { provider.destroy(); }
}

if (require.main === module) {
  if (process.argv.includes("--compile")) {
    const a = compile();
    process.stdout.write(JSON.stringify({compiler: a.compiler, bytecode_bytes: (a.bytecode.length - 2) / 2}) + "\n");
  } else {
    const input = JSON.parse(fs.readFileSync(0, "utf8"));
    run(input).then(result => process.stdout.write(JSON.stringify(result) + "\n"))
      .catch(error => { process.stderr.write((error.shortMessage || error.message) + "\n"); process.exitCode = 1; });
  }
}
module.exports = {compile, run, statuses};
