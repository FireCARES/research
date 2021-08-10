Unit Optimizer
---

This is a tool written to allow the analysis of unit placement across a jurisdiction, using their existing stations and their historic unit allocations to stations, with the goal of approximating the best station to either add or remove a unit, along with estimating that addition or removal's impact on overall department response metrics. This tool does not replace a more thorough analysis of a department's apparatus allocations, nor does it provide any information beyond an approximation using historic department data. The output of this tool may be used to aid in informing a department's decision, but the data it provides is in no way certified, confirmed, or assumed to be correct in any way shape or form, and should never be used as the basis for a department's apparatus allocations.

This tool is provided for free use to anyone who can access the backend databases required to populate its analyses. Its performance is not verified. Use at your own risk!

# Dependencies
pure python dependencies are included in a conda `environment.yml` file.

You will need a postgresql service definition that can connect to the firecares database, and will want to edit the `SERVICE` global constant in `unit_decision.py` to reflect that service definition.

You will need access to the NFORS elasticsearch service. Typically this is accomplished via a white-listed VPN coupled with an appropriate kubernetes config and port-forwarding of the appropriate pods.

# Running the tool

See the `delray_unit_optimizer` python notebook for notes on how to run this tool.