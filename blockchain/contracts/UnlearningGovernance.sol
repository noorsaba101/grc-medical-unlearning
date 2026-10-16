// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice Local PoC: trusted actors attest off-chain work; no ML runs in the EVM.
contract UnlearningGovernance {
    enum Status { NONE, REGISTERED, AUTHORIZED, VERIFIED_PASS, VERIFIED_FAIL, COMPLETED }
    struct Request {
        bytes32 requestHash;
        bytes32 policyIdHash;
        bytes32 policyHash;
        bytes32 sourceModelHash;
        bytes32 resultModelHash;
        bytes32 verificationHash;
        Status status;
        uint64 registeredAt;
        uint64 authorizedAt;
        uint64 verifiedAt;
        uint64 completedAt;
        address requester;
        address authorizer;
        address verifiedBy;
        address completedBy;
    }
    address public immutable governor;
    address public immutable operator;
    address public immutable verifier;
    mapping(bytes32 => Request) private requests;
    event StateChanged(bytes32 indexed requestId, address indexed actor, Status status, uint256 timestamp);

    constructor(address operator_, address verifier_) {
        require(operator_ != address(0) && verifier_ != address(0), "zero actor");
        require(operator_ != msg.sender && verifier_ != msg.sender && operator_ != verifier_, "distinct actors required");
        governor = msg.sender;
        operator = operator_;
        verifier = verifier_;
    }
    modifier onlyGovernor() { require(msg.sender == governor, "governor only"); _; }
    modifier onlyOperator() { require(msg.sender == operator, "operator only"); _; }
    modifier onlyVerifier() { require(msg.sender == verifier, "verifier only"); _; }

    function registerUnlearningRequest(bytes32 id, bytes32 requestHash, bytes32 policyIdHash,
                                        bytes32 policyHash, bytes32 sourceModelHash) external onlyOperator {
        require(id != bytes32(0) && requestHash != bytes32(0) && policyIdHash != bytes32(0)
                && policyHash != bytes32(0) && sourceModelHash != bytes32(0), "zero hash");
        Request storage r = requests[id];
        require(r.status == Status.NONE, "already registered");
        r.requestHash = requestHash;
        r.policyIdHash = policyIdHash;
        r.policyHash = policyHash;
        r.sourceModelHash = sourceModelHash;
        r.status = Status.REGISTERED;
        r.registeredAt = uint64(block.timestamp);
        r.requester = msg.sender;
        emit StateChanged(id, msg.sender, r.status, block.timestamp);
    }
    function authorizeRequest(bytes32 id) external onlyGovernor {
        Request storage r = requests[id];
        require(r.status == Status.REGISTERED, "must be registered");
        r.status = Status.AUTHORIZED;
        r.authorizedAt = uint64(block.timestamp);
        r.authorizer = msg.sender;
        emit StateChanged(id, msg.sender, r.status, block.timestamp);
    }
    function recordVerification(bytes32 id, bytes32 resultHash, bytes32 verificationHash, bool passed) external onlyVerifier {
        Request storage r = requests[id];
        require(r.status == Status.AUTHORIZED, "must be authorized");
        require(resultHash != bytes32(0) && verificationHash != bytes32(0), "zero hash");
        r.resultModelHash = resultHash;
        r.verificationHash = verificationHash;
        r.status = passed ? Status.VERIFIED_PASS : Status.VERIFIED_FAIL;
        r.verifiedAt = uint64(block.timestamp);
        r.verifiedBy = msg.sender;
        emit StateChanged(id, msg.sender, r.status, block.timestamp);
    }
    function completeRequest(bytes32 id) external onlyGovernor {
        Request storage r = requests[id];
        require(r.status == Status.VERIFIED_PASS, "verification must pass");
        r.status = Status.COMPLETED;
        r.completedAt = uint64(block.timestamp);
        r.completedBy = msg.sender;
        emit StateChanged(id, msg.sender, r.status, block.timestamp);
    }
    function getRequest(bytes32 id) external view returns (Request memory) { return requests[id]; }
}
