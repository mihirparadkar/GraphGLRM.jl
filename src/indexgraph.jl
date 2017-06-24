#export IndexGraph

#using LightGraphs
"""
The data structure that stores a graph and a mapping from indices to their
representations in the graph
"""
mutable struct IndexGraph{T}
  idx::Dict{T,Int64}
  graph::Graph
end

function IndexGraph{T}(nodes::AbstractArray{T}, edges::AbstractArray{Tuple{T,T}})
  idx = Dict{T, Int64}(n => i for (i,n) in enumerate(nodes))
  graph = Graph(length(idx))
  for e in edges
    if !haskey(idx, e[1]) || !haskey(idx,e[2])
      error("Edges may only contain nodes in the graph")
    else
      add_edge!(graph, idx[e[1]], idx[e[2]])
    end
  end
  IndexGraph(idx, graph)
end

function IndexGraph{T}(nodes::AbstractArray{T})
  idx = Dict{T, Int64}(n => i for (i,n) in enumerate(nodes))
  graph = Graph(length(idx))
  IndexGraph(idx, graph)
end

function IndexGraph(g::Graph)
  idx = Dict{Int64, Int64}(i => i for i in 1:nv(g))
  graph = g
  IndexGraph(idx, graph)
end

"""
Helper function to embed an index graph into multidimensional losses.
"""
function embed_graph(ig::IndexGraph, yidxs::Array)
  nodes = 1:yidxs[end][end]
  newedges = Tuple{Int64,Int64}[]
  for edge in edges(ig.graph)
    push!(newedges, (yidxs[edge.src], yidxs[edge.dst]))
  end
  IndexGraph(nodes, newedges)
end
