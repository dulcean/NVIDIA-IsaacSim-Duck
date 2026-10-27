* Add opt-in visual volleyball PPO player exchange and policy-consistency loss
  to share receiving experience across robot identities. Preserve existing model
  and optimizer state on resume, persist the enabled method, and restart the
  stagnation window only on its first explicit activation. Physical held-out
  success gates remain required; skill transfer requires GPU validation.
