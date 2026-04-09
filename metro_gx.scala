import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.util.hashing.MurmurHash3

val datasetPath = sys.env.getOrElse("METRO_CSV_PATH", "metro.csv")
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(datasetPath)

def normalizeStation(name: String): String =
  name.replaceAll("\\[.*?\\]", "").replaceAll("\\s+", " ").trim.toLowerCase

def vertexIdFromKey(key: String): VertexId =
  (MurmurHash3.stringHash(key).toLong & 0x7fffffffffffffffL)

val cleanDF = df
  .select(
    "`Station Names`",
    "`Metro Line`",
    "`Dist. From First Station(km)`"
  )
  .na.drop()

val stationRows = cleanDF.select("`Station Names`").distinct().collect()
val vertices: RDD[(VertexId, String)] = sc.parallelize(
  stationRows.map { row =>
    val stationName = row.getString(0)
    val stationKey = normalizeStation(stationName)
    (vertexIdFromKey(stationKey), stationName)
  }
)

val lineRows = cleanDF.collect().groupBy(_.getAs[String]("Metro Line"))
val edgeList = lineRows.values.flatMap { rows =>
  rows
    .sortBy(_.getAs[Double]("Dist. From First Station(km)"))
    .sliding(2)
    .collect {
      case Array(a, b) =>
        val src = vertexIdFromKey(normalizeStation(a.getAs[String]("Station Names")))
        val dst = vertexIdFromKey(normalizeStation(b.getAs[String]("Station Names")))
        Edge(src, dst, 1)
    }
}.toList

val edges: RDD[Edge[Int]] = sc.parallelize(edgeList)

val graph = Graph(vertices, edges).cache()

val deg = graph.degrees
val inDeg = graph.inDegrees
val outDeg = graph.outDegrees

println("\n===== DEGREE CENTRALITY =====")
deg.join(vertices).collect().foreach { case (_, (degree, name)) =>
  println(s"$name -> degree=$degree")
}

println("\n===== BUSIEST =====")
deg.join(vertices).sortBy(_._2._1, ascending = false).take(10).foreach {
  case (_, (degree, name)) => println(s"$name -> degree=$degree")
}

println("\n===== LEAST CONNECTED =====")
deg.join(vertices).sortBy(_._2._1, ascending = true).take(10).foreach {
  case (_, (degree, name)) => println(s"$name -> degree=$degree")
}

println("\n===== ISOLATED STATIONS =====")
val isolated = graph.vertices.leftOuterJoin(deg).filter { case (_, (_, d)) => d.isEmpty }
if (isolated.isEmpty()) {
  println("No isolated stations found in the constructed network.")
} else {
  isolated.collect().foreach { case (_, (name, _)) => println(name) }
}

val ranks = graph.pageRank(0.0001).vertices

println("\n===== PAGERANK =====")
ranks.join(vertices).sortBy(_._2._1, ascending = false).take(15).foreach {
  case (_, (rank, name)) => println(f"$name -> PageRank=$rank%.6f")
}

println("\n===== IN/OUT DEGREE SNAPSHOT =====")
inDeg.join(vertices).sortBy(_._2._1, ascending = false).take(5).foreach {
  case (_, (value, name)) => println(s"$name -> inDegree=$value")
}
outDeg.join(vertices).sortBy(_._2._1, ascending = false).take(5).foreach {
  case (_, (value, name)) => println(s"$name -> outDegree=$value")
}