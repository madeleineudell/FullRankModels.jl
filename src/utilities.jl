import LowRankModels.copy_estimate

export copy_estimate

function copy_estimate(g::GFRM)
  return GFRM(g.A,g.losses,g.r,g.k,
              g.observed_features,g.observed_examples,
              copy(g.U),copy(g.W))
end