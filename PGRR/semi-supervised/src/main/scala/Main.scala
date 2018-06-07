import breeze.linalg._
import breeze.numerics._
import org.apache.spark.graphx._
import com.aliyun.odps.TableSchema
import com.aliyun.odps.data.Record
import org.apache.spark.odps.OdpsOps
import org.apache.spark.{SparkConf, SparkContext}

object Main {
  val inputSemiTable = "sample_semi"
  val inputNodeTable = "sample_feature"
  val inputEdgeTable = "sample_edge"
  val outputClusterTable = "sample_label"
  val featureDimension = 100
  var delta = 1.0
  var minError = 0.0
  val rho = 1.0
  val lambda = 1e-5
  val lambda2 = 1e-5

  def sqr(x: Double): Double = {
    return x * x
  }

  def computeLoss(graph: Graph[
    (DenseVector[Double], DenseVector[Double], Long, Long, Long),
    (Double, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])
    ]): Double = {
    var loss: Double = 0
    loss += graph.vertices.map { case (_, (xi, ci, _, _,_)) => 0.0 }.sum
    loss += lambda * graph.triplets.map(triplet => {
      0.0
    }).sum
    return loss
  }


  def updateDelta(graph: Graph[
    (DenseVector[Double], DenseVector[Double], Long, Long, Long),
    (Double, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])
    ]): Int = {

    val tmp1 = graph.triplets.map(triplet => {
      val ci = triplet.srcAttr._2
      val cj = triplet.dstAttr._2
      val wij = exp(-norm(ci - cj))
      val ti = triplet.srcAttr._5
      val tj = triplet.dstAttr._5
      var flag = 0
      if (ti == -1 || tj == -1) flag = -1
      else if (ti == tj) flag = 1
      else flag = 0
      (wij, flag)
    })
    var array = tmp1.filter(_._2 != -1).collect
    array = array.sortWith(_._1 < _._1)
    var cnt0 = 0
    var cnt1 = 0
    for (elem <- array) {
      if (elem._2 == 1) cnt1 += 1
      else cnt0 +=1
    }
    delta = 1.0
    minError = cnt0 + cnt1
    var cur0 = 0
    var cur1 = 0
    for (elem <- array) {
      if (elem._2 == 1) cur1 += 1
      else cur0 += 1
      val err = cur1 + (cnt0 - cur0)
      if (err <= minError) {
        minError = err
        delta = elem._1
      }
    }
    return 0
  }

  def updateC(oldGraph: Graph[
    (DenseVector[Double], DenseVector[Double], Long, Long, Long),
    (Double, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])
    ]): Graph[
    (DenseVector[Double], DenseVector[Double], Long, Long, Long),
    (Double, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])
    ] = {

    var graph = oldGraph
    graph = graph.mapTriplets(triplet => {
      val xi = triplet.srcAttr._1
      val xj = triplet.dstAttr._1
      val pi = triplet.srcAttr._4
      val pj = triplet.dstAttr._4
      var sij = 0.0
      if (pi == pj) sij = 1.0
      else sij = exp(-norm(xi-xj))
      (sij, triplet.attr._2, triplet.attr._3, triplet.attr._4, triplet.attr._5)
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
        val pi = oldValue._4
        val ti = oldValue._5
        val ci = (xi * 2.0 + extraValue * rho) / (2.0 + rho * di + lambda2 * 2)
        (xi, ci, di, pi, ti)
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
      println(" iteration " + iteration + ":")
      println(" loss = " + computeLoss(graph))

      prevG.unpersistVertices(blocking = false)
      prevG.edges.unpersist(blocking = false)
    }

    return graph
  }

  def main(args: Array[String]): Unit = {
    val projectName = "da_intern_dev"
    val conf = new SparkConf().setAppName("PGRR-SEMI")
    val sc = new SparkContext(conf)
    val odpsOps = new OdpsOps(sc)

    //(id, (feature, clusterCenter, degree, clusterIndex, semi_label))
    //(id, (xi, ci, di, pi, ti))
    var node = odpsOps.readTable[(Long, (DenseVector[Double], DenseVector[Double], Long, Long, Long))](
      projectName,
      inputNodeTable,
      (r: Record, schema: TableSchema) => {
        val vectorBuilder = new VectorBuilder[Double](featureDimension)
        for (i <- 1 to featureDimension) {
          vectorBuilder.add(i - 1, r.getDouble(i))
        }
        (r.getBigint(0).toLong, (vectorBuilder.toDenseVector, vectorBuilder.toDenseVector, 0L, 0L, -1L))
      }, 0).cache()

    //(srcId, dstId, (sij, zij, zji, uij, uji))
    var edge = odpsOps.readTable[Edge[(Double, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])]](
      projectName,
      inputEdgeTable,
      (r: Record, schema: TableSchema) => {
        Edge(r.getBigint(0).toLong, r.getBigint(1).toLong,
          (0.0,
            DenseVector.zeros[Double](featureDimension),
            DenseVector.zeros[Double](featureDimension),
            DenseVector.zeros[Double](featureDimension),
            DenseVector.zeros[Double](featureDimension)
          )
        )
      }, 0).cache()

    var graph = Graph(node, edge).cache()
    graph = graph.joinVertices(graph.degrees)((_, oldValue, di) => {
      val xi = oldValue._1
      val ci = oldValue._2
      (xi, ci, di, oldValue._4, oldValue._5)
    })

    var semi = odpsOps.readTable[(Long, Long)](
      projectName,
      inputSemiTable,
      (r: Record, schema: TableSchema) => {
        (r.getBigint(0).toLong, r.getBigint(1).toLong)
      }
    )
    graph = graph.joinVertices(semi)((_, oldValue, ti) => {
      (oldValue._1, oldValue._2, oldValue._3, oldValue._4, ti)
    })

    graph = graph.mapTriplets(triplet => {
      val xi = triplet.srcAttr._1
      val xj = triplet.dstAttr._1
      (exp(-norm(xi-xj)), xi, xj, DenseVector.zeros[Double](featureDimension), DenseVector.zeros[Double](featureDimension))
    })

    for (iter <- 1 to 50) {
      updateDelta(graph)
      val tmpGraph = graph.mapTriplets(triplet => exp(-norm(triplet.srcAttr._2 - triplet.dstAttr._2)))
      val subGraph = tmpGraph.subgraph(epred = edge => edge.attr >= delta)
      val clusterResult = subGraph.connectedComponents().vertices
      graph = graph.joinVertices(clusterResult)((_, oldValue, pi) => {
        val xi = oldValue._1
        val ci = oldValue._2
        val di = oldValue._3
        val ti = oldValue._5
        (xi, ci, di, pi, ti)
      }
      )
      val loss = minError + computeLoss(graph)
      println("" + iter + ": " + loss)
      graph = updateC(graph)
    }

    val clusterResult = graph.vertices.map{case (id, (xi, ci, di, pi,ti)) => (id, pi)}

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