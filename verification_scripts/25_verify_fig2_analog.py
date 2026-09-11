# Check script 13's eval multiplier against the real ansatz, off convergence_ansatzes.png.
cobyla_max_epoch = 200
spsa_max_epoch = 455     # curve runs slightly past the 400 gridline
nft_max_epoch = 405
configured_maxiter = 200  # ANSATZ_MAXITER

print(f"Cobyla: {cobyla_max_epoch} evals / {configured_maxiter} configured = "
      f"{cobyla_max_epoch/configured_maxiter:.2f}x")
print(f"SPSA:   {spsa_max_epoch} evals / {configured_maxiter} configured = "
      f"{spsa_max_epoch/configured_maxiter:.2f}x   (toy-example prediction from script 13 was 3.02x)")
print(f"NFT:    {nft_max_epoch} evals / {configured_maxiter} configured = "
      f"{nft_max_epoch/configured_maxiter:.2f}x   (toy-example prediction from script 13 was 2.06x)")
