import breeze.linalg._
import breeze.numerics._
import org.apache.spark.graphx._
import com.aliyun.odps.TableSchema
import com.aliyun.odps.data.Record
import org.apache.spark.odps.OdpsOps
import org.apache.spark.{SparkConf, SparkContext}

object Main {
  val inputNodeTable = "sample_feature"
  val inputEdgeTable = "sample_edge"
  val outputClusterTable = "sample_label"

  val featureDimension = 100
  val lambda = 1e-5
  val delta = 1.0
  val rho = 1.0
  val lambda2 = 1e5

  def sqr(x: Double): Double = {
    return x * x
  }

  def computeLoss(graph: Graph[
    (DenseVector[Double], DenseVector[Double], Long),
    (Double, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])
    ]): Double = {
    var loss: Double = 0
    loss += graph.vertices.map { case (_, (xi, ci, _)) => sqr(norm(xi - ci))+lambda2*sqr(norm(ci)) }.sum
    loss += lambda * graph.triplets.map(triplet => {
      val sij = triplet.attr._1
      val ci = triplet.srcAttr._2
      val cj = triplet.dstAttr._2
      norm(ci - cj) * sij
    }).sum
    return loss
  }

  def main(args: Array[String]): Unit = {
    val projectName = "da_intern_dev"
    val conf = new SparkConf().setAppName("PGRR-unsupervised")
    val sc = new SparkContext(conf)
    val odpsOps = new OdpsOps(sc)

    var node = odpsOps.readTable[(Long, (DenseVector[Double], DenseVector[Double], Long))](
      projectName,
      inputNodeTable,
      (r: Record, schema: TableSchema) => {
        val vectorBuilder = new VectorBuilder[Double](featureDimension)
        for (i <- 1 to featureDimension) {
          vectorBuilder.add(i - 1, r.getDouble(i))
        }
        (r.getBigint(0).toLong, (vectorBuilder.toDenseVector, vectorBuilder.toDenseVector, 0L))
      }, 10)
    node = node.repartition(10)

    var inputEdge = odpsOps.readTable[(Long, Long)](projectName, inputEdgeTable,
      (r: Record, schema: TableSchema) =>
        (r.getBigint(0).toLong, r.getBigint(1).toLong),10)
    inputEdge = inputEdge.repartition(10)

    val edge = inputEdge.map { case (src, dst) => Edge(src, dst,
      (0.0,
        DenseVector.zeros[Double](featureDimension),
        DenseVector.zeros[Double](featureDimension),
        DenseVector.zeros[Double](featureDimension),
        DenseVector.zeros[Double](featureDimension)
      )
    )
    }

    var graph = Graph(node, edge)
    graph = graph.joinVertices(graph.degrees)((_, oldValue, di) => {
      val xi = oldValue._1
      val ci = oldValue._2
      (xi, ci, di)
    })

    graph = graph.mapTriplets(triplets => {
      val xi = triplets.srcAttr._1
      val xj = triplets.dstAttr._1
      val di = triplets.srcAttr._3
      val dj = triplets.dstAttr._3
      var oi = 0.0
      if (di > 10) oi = 0.1
      else if (di < 5) oi = 1
      else oi = 0.5
      var oj = 0.0
      if (dj > 10) oj = 0.1
      else if (dj < 5) oj = 1
      else oj = 0.5
      val sij = oi * oj * exp(norm(xi - xj) * (-1.0))
      (sij, xi, xj, triplets.attr._4, triplets.attr._5)
    })

    for (iteration <- 1 to 50) {
      val prevG = graph

      //c-update
      val tempValue = graph.aggregateMessages[DenseVector[Double]](
        triplet => {
          val zij = triplet.attr._2
          val zji = triplet.attr._3
          val uij = triplet.attr._4
          val uji = triplet.attr._5
          triplet.sendToSrc(zij - uij)
          triplet.sendToDst(zji - uji)
        },
        (a, b) => a + b
      )
      graph = graph.joinVertices(tempValue)((_, oldValue, extraValue) => {
        val xi = oldValue._1
        val di = oldValue._3
        val ci = (xi * 2.0 + extraValue * rho) / (2.0 + rho * di + lambda2 * 2)
        (xi, ci, di)
      })

      //z-update
      graph = graph.mapTriplets(triplets => {
        val sij = triplets.attr._1
        val uij = triplets.attr._4
        val uji = triplets.attr._5
        val ci = triplets.srcAttr._2
        val cj = triplets.dstAttr._2
        var theta = 0.0
        if (norm(ci + uij - (cj + uji)) == 0.0) theta = 0.5
        else theta = max(1 - lambda * sij / (rho * norm(ci + uij - (cj + uji))), 0.5)
        val zij = (ci + uij) * theta + (cj + uji) * (1 - theta)
        val zji = (ci + uij) * (1 - theta) + (cj + uji) * theta
        (sij, zij, zji, uij, uji)
      })

      //u-update
      graph = graph.mapTriplets(triplets => {
        val sij = triplets.attr._1
        val zij = triplets.attr._2
        val zji = triplets.attr._3
        val uij = triplets.attr._4
        val uji = triplets.attr._5
        val ci = triplets.srcAttr._2
        val cj = triplets.dstAttr._2
        (sij, zij, zji, uij + ci - zij, uji + cj - zji)
      })

      graph.cache()
      println("iteration " + iteration + ":")
      println("loss = " + computeLoss(graph))
      val newGraph = graph.mapTriplets(triplest => exp(-norm(triplest.srcAttr._2 - triplest.dstAttr._2)))
      val subGraph = newGraph.subgraph(epred = edge => edge.attr >= delta)
      println("cluster number = " + subGraph.connectedComponents().vertices.map {
        case (_, cluster) => cluster
      }.distinct.count)

      prevG.unpersistVertices(blocking = false)
      prevG.edges.unpersist(blocking = false)
    }

    val newGraph = graph.mapTriplets(triplest => exp(-norm(triplest.srcAttr._2 - triplest.dstAttr._2)))
    val subGraph = newGraph.subgraph(epred = edge => edge.attr >= delta)
    val clusterResult = subGraph.connectedComponents().vertices

    odpsOps.saveToTable(projectName, outputClusterTable, clusterResult,
      (v: (Long, Long), r: Record, schema: TableSchema) => {
        r.set(0, v._1)
        r.set(1, v._2)
      },
      isOverWrite = true
    )

    sc.stop()
  }
}

